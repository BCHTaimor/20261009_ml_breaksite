#!/usr/bin/env python3
"""Train a 3-input count network with configurable depth on shared TSVs.

Use --help for options. Poisson count loss requires raw integer HTGTS counts;
normalized TLX targets must be resolved before use. CPU execution is seeded.
Test labels are evaluated only after validation selects the best epoch.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_data import load_datasets, load_regions

ARCHITECTURE = [3, 64, 64, 32, 16, 8, 1]


class BreakCountNet(nn.Module):
    """Map three standardized genomic features to expected break counts.

    AF: layers represents a count regressor described by architecture.
    Invariant: the requested number of ReLU hidden layers precedes one Softplus output;
    feature order is OK-seq, log1p(GRO-seq), log(bin size in bp).
    Mutable state: layers and its parameters are exposed through PyTorch's
    module API; training and state loading intentionally mutate them.
    """
    def __init__(self, hidden_layers=5):
        """Initialize the count regressor using the current PyTorch RNG state.

        @param hidden_layers: Positive integer depth. Five preserves widths
            [64, 64, 32, 16, 8]; deeper models prepend width-64 layers (30 gives
            27 width-64 layers followed by 32, 16, 8). Shallower models use the
            first hidden_layers widths of the original five-layer network.
        @returns: None. Registers randomly initialized layers on this instance
            and advances the PyTorch RNG; does not set the random seed.
        @raises ValueError: If hidden_layers is not a positive integer.
        """
        super().__init__()
        if isinstance(hidden_layers, bool) or not isinstance(hidden_layers, int) or hidden_layers < 1:
            raise ValueError('Hidden layers must be a positive integer')
        widths = [64] * max(0, hidden_layers - 5) + ARCHITECTURE[1:-1][:hidden_layers]
        self.architecture = [3, *widths, 1]
        layers = []
        for left, right in zip(self.architecture[:-2], self.architecture[1:-1]):
            layers.extend([nn.Linear(left, right), nn.ReLU()])
        layers.extend([nn.Linear(self.architecture[-2], 1), nn.Softplus()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Evaluate the network, preserving autograd when it is enabled.

        @param x: Floating torch.Tensor of shape (N, 3), containing standardized
            features in the class-defined order; dtype/device must match layers.
        @returns: Tensor of shape (N,) with the model's dtype/device, containing
            expected break counts (Softplus output plus 1e-6). Positivity assumes
            finite computation; this method does not validate finite outputs.
        @raises RuntimeError: If tensor shape, dtype, or device is incompatible.
        No parameters or input values are modified.
        """
        return self.layers(x).squeeze(-1) + 1e-6


class Preprocessor:
    """Represent feature transforms and scaling learned from training rows.

    AF: mean and scale standardize [OK-seq, log1p(GRO-seq), log(bin bp)].
    Invariant after fit on valid, numerically stable input: both attributes are
    length-3 float64 arrays and scale is positive, with near-constant columns
    assigned scale 1. A new instance is unfitted and has neither attribute.
    Mutable state: fit replaces mean/scale; these public arrays are not protected
    from external mutation. state returns detached lists for serialization.
    """
    @staticmethod
    def raw_transform(x):
        """Apply logarithms to GRO-seq and bin size, leaving OK-seq unchanged.

        @param x: Numeric array-like of shape (N, 3), ordered as OK-seq, GRO-seq,
            bin size in bp; all values finite, GRO-seq >= 0 and bin size > 0.
        @returns: New float64 ndarray of shape (N, 3) containing OK-seq,
            log1p(GRO-seq), and log(bin size). Empty (0, 3) input is permitted.
        @raises ValueError: If conversion, shape, finiteness, or domain is invalid.
        @raises TypeError: If values cannot be converted to numeric array data.
        Does not mutate the input or fit any scaling state.
        """
        x = np.asarray(x, dtype=np.float64).copy()
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError('Expected finite Nx3 input')
        if np.any(x[:, 1] < 0) or np.any(x[:, 2] <= 0):
            raise ValueError('GRO-seq must be nonnegative and bin size positive')
        x[:, 1] = np.log1p(x[:, 1]); x[:, 2] = np.log(x[:, 2])
        return x

    def fit(self, x):
        """Fit column means and population standard deviations on training rows.

        @param x: Nonempty numeric array-like of shape (N, 3), in raw feature
            units and satisfying raw_transform's domain; supply training only.
        @returns: This fitted Preprocessor instance. Replaces mean and scale
            with length-3 float64 arrays; standard deviations < 1e-12 become 1.
        @raises ValueError: If input is empty or violates raw_transform's contract.
        @raises TypeError: If input cannot be converted to numeric array data.
        Input values are not mutated; fitting again replaces prior statistics.
        """
        z = self.raw_transform(x)
        if len(z) == 0:
            raise ValueError('Cannot fit empty training data')
        self.mean = z.mean(axis=0)
        self.scale = z.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def transform(self, x):
        """Apply logarithms and previously fitted training statistics.

        @param x: Numeric array-like of shape (N, 3), in raw feature units and
            satisfying raw_transform's domain, including empty (0, 3) arrays.
        @returns: New finite float32 ndarray of shape (N, 3), standardized using
            this instance's mean and scale; input and fitted state are unchanged.
        @raises AttributeError: If fit has not established mean and scale.
        @raises ValueError: If input is invalid or transformed values are nonfinite.
        @raises TypeError: If input cannot be converted to numeric array data.
        """
        z = ((self.raw_transform(x) - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite transformed features')
        return z

    def state(self):
        """Export fitted statistics without exposing their underlying arrays.

        @returns: New dict with 'mean' and 'scale', each a length-3 list of
            Python floats; modifying these lists does not change fitted state.
        @raises AttributeError: If fit has not established mean and scale.
        """
        return {'mean':self.mean.tolist(), 'scale':self.scale.tolist()}


def predict(model, x, batch_size):
    """Predict counts in bounded CPU batches, with autograd disabled.

    @param model: CPU BreakCountNet (or compatible module) accepting (N, 3)
        tensors and returning one count per row; dtype must match x.
    @param x: Nonempty float32 ndarray of shape (N, 3), already standardized.
    @param batch_size: Positive integer maximum number of rows per forward pass.
    @returns: Concatenated ndarray of shape (N,) in input order, with the model's
        output dtype (normally float32); prediction validity is not checked here.
    @raises ValueError: If x is empty or batch_size is zero.
    @raises RuntimeError: If model and tensor shape, dtype, or device conflict.
    Sets model to evaluation mode and leaves it there; does not update weights.
    """
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch_size])).numpy()
                               for i in range(0, len(x), batch_size)])


def metrics(y, predicted):
    """Calculate equally weighted count errors across corresponding bin rows.

    @param y: Nonempty finite nonnegative numeric ndarray of shape (N,), holding
        observed counts. Caller must validate this truth array.
    @param predicted: Finite strictly positive numeric ndarray of shape (N,),
        holding expected counts paired with y in the same order.
    @returns: Dict of Python floats under 'mae', 'rmse', and 'poisson_deviance';
        deviance is averaged over rows and handles y == 0 without taking log(0).
    @raises ValueError: If predictions are nonfinite/nonpositive or arrays cannot
        broadcast. Inputs are not mutated.
    """
    if not np.isfinite(predicted).all() or np.any(predicted <= 0):
        raise ValueError('Model produced invalid predictions')
    term = predicted - y
    positive = y > 0
    term[positive] += y[positive] * np.log(y[positive] / predicted[positive])
    return {'mae':float(np.mean(np.abs(y-predicted))),
            'rmse':float(np.sqrt(np.mean((y-predicted)**2))),
            'poisson_deviance':float(2*np.mean(term))}


def file_hash(path):
    """Stream a file through SHA-256 in 1 MiB chunks.

    @param path: str or Path naming a readable file whose bytes are to be hashed.
    @returns: 64-character hexadecimal SHA-256 digest of the complete file.
    @raises OSError: If the file cannot be opened or read.
    Reads and closes the file without changing its contents.
    """
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    """Train on shared TSVs, select a validation epoch, and evaluate held-out bins.

    Command-line inputs are read from sys.argv (there are no Python parameters):
    --dataset paths supply raw-count TSVs; --split-dir supplies region BEDs and
    manifest; --target-kind raw-count and --boundary-buffer-bp are required.
    --hidden-layers selects positive integer depth (default 5; see BreakCountNet).
    Optional flags configure validation, optimization, seeds, output directory,
    and prediction export; --help lists their defaults. Preprocessing and the
    rate baseline are fitted only on training rows. Validation selects weights
    before test labels are evaluated; supplied bin resolutions share one model.

    @returns: None. Creates a new output directory containing config.json,
        metrics.tsv, history.tsv, validation_regions.bed, model/model.pt, and
        optionally predictions.tsv; prints completion to stdout. Source files
        are read only. Failed writes can leave a partially populated run folder.
    @raises SystemExit: argparse exits for --help, invalid arguments, or caught
        ValueError/OSError/KeyError failures (including an existing output path).
    @raises RuntimeError: Unhandled PyTorch execution or serialization failure.
    Side effects: seeds global Python/NumPy/PyTorch RNGs, sets PyTorch CPU threads
    to one and deterministic algorithms on, and advances RNG state in training.
    These process-wide settings are not restored on return or failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--target-kind', choices=['raw-count'], required=True)
    parser.add_argument('--boundary-buffer-bp', type=int, required=True,
                        help='Trim this many bp on both sides of each partition transition')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--hidden-layers', type=int, default=5,
                        help='Positive hidden-layer count; depths above 5 add width-64 layers')
    parser.add_argument('--validation-seed', type=int, default=123)
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.output_dir = Path('results') / f'run_{stamp}_{secrets.token_hex(3)}'
    try:
        if args.hidden_layers < 1:
            raise ValueError('Hidden layers must be a positive integer')
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError('Output directory already exists; refusing overwrite')
        if min(args.epochs,args.patience,args.batch_size) < 1 or args.boundary_buffer_bp < 0:
            raise ValueError('Epochs/patience/batch size must be positive and buffer nonnegative')
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError('Learning rate must be finite and positive')
        split_manifest = json.loads((args.split_dir/'manifest.json').read_text())
        regions = load_regions(args.split_dir, args.validation_fraction, args.validation_seed)
        data = load_datasets(args.dataset, regions, args.boundary_buffer_bp)
        parts, raw, y = data['partition'], data['x'], data['y']
        masks = {part:parts == part for part in ('train','validation','test')}
        for part, mask in masks.items():
            if not mask.any():
                raise ValueError(f'No usable {part} bins after region/buffer filtering')
        if not y[masks['train']].sum():
            raise ValueError('Training bins contain no observed breaks')
        for size in np.unique(raw[:,2]):
            for part, mask in masks.items():
                if not np.any(mask & (raw[:,2] == size)):
                    raise ValueError(f'Bin size {size:g} has no usable {part} rows')
        random.seed(args.model_seed); np.random.seed(args.model_seed % (2**32))
        torch.manual_seed(args.model_seed); torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        scaler = Preprocessor().fit(raw[masks['train']])
        x = scaler.transform(raw)
        model = BreakCountNet(args.hidden_layers)
        loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        train_x = torch.from_numpy(x[masks['train']])
        train_y = torch.from_numpy(y[masks['train']].astype(np.float32))
        loader = DataLoader(TensorDataset(train_x,train_y), batch_size=args.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(args.model_seed))
        best_loss, best_state, best_epoch, stale, history = float('inf'), None, 0, 0, []
        for epoch in range(1,args.epochs+1):
            model.train(); total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(bx), by)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                total_loss += loss.item()*len(bx)
            val_pred = predict(model,x[masks['validation']],args.batch_size)
            val_y = y[masks['validation']]
            val_loss = float(np.mean(val_pred-val_y*np.log(val_pred+1e-8)))
            if not math.isfinite(val_loss):
                raise ValueError('Nonfinite validation loss')
            history.append((epoch,total_loss/len(train_x),val_loss))
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        model.load_state_dict(best_state)
        predictions = {part:predict(model,x[mask],args.batch_size) for part,mask in masks.items()}
        # Rate baseline learns only from fitting rows and scales with bin size.
        rate = float(y[masks['train']].sum()/raw[masks['train'],2].sum())
        metric_rows = []
        for part,mask in masks.items():
            sizes = raw[mask,2]; truth = y[mask]
            for size in [None, *np.unique(sizes)]:
                subset = np.ones(len(sizes),dtype=bool) if size is None else sizes == size
                for name, pred in [('network',predictions[part]),('constant_rate',rate*sizes)]:
                    values = metrics(truth[subset],np.maximum(pred[subset],1e-6))
                    metric_rows.append({'partition':part,'bin_size_bp':'all' if size is None else int(size),
                                        'model':name,'n':int(subset.sum()),**values})
        config = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
                  'arguments':{k:[str(p) for p in v] if k=='dataset' else str(v) if isinstance(v,Path) else v
                               for k,v in vars(args).items()},
                  'architecture':model.architecture,'feature_order':['okseq','log1p_groseq','log_bin_size_bp'],
                  'target':'raw observed HTGTS break count','loss':'PoissonNLLLoss(log_input=False,full=False)',
                  'best_epoch':best_epoch,'epochs_completed':len(history),'preprocessor':scaler.state(),
                  'row_counts':{p:int((parts==p).sum()) for p in (*masks,'excluded')},
                  'training_count_rate_per_bp':rate,'split_seed':split_manifest.get('config',{}).get('seed'),
                  'inputs':{str(p.resolve()):file_hash(p) for p in [*args.dataset,
                            args.split_dir/'manifest.json',args.split_dir/'train_regions.bed',args.split_dir/'test_regions.bed']},
                  'code':{p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('ml_data.py')]},
                  'versions':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__},
                  'bin_rule':'Full fixed-size zero-anchored bins; full containment after transition buffers',
                  'validation_rule':'Whole original training region groups, selected by validation seed',
                  'row_weighting':'Equal weight per row across all supplied resolutions; overlapping resolutions are correlated'}
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/'model').mkdir()
        torch.save({'state_dict':best_state,'architecture':model.architecture,'preprocessor':scaler.state()},
                   args.output_dir/'model/model.pt')
        (args.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        with (args.output_dir/'metrics.tsv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(metric_rows[0]),delimiter='\t')
            writer.writeheader(); writer.writerows(metric_rows)
        with (args.output_dir/'history.tsv').open('w') as handle:
            writer=csv.writer(handle,delimiter='\t'); writer.writerow(['epoch','train_loss','validation_loss'])
            writer.writerows(history)
        with (args.output_dir/'validation_regions.bed').open('w') as handle:
            for chrom,start,end,label in regions:
                if label=='validation': handle.write(f'{chrom}\t{start}\t{end}\n')
        if args.save_predictions:
            with (args.output_dir/'predictions.tsv').open('w') as handle:
                writer=csv.writer(handle,delimiter='\t'); writer.writerow(['bin_id','partition','predicted_break_count'])
                for part,mask in masks.items():
                    for i,pred in zip(np.flatnonzero(mask),predictions[part]):
                        writer.writerow([data['rows'][i]['bin_id'],part,float(pred)])
        print(f'Wrote {args.output_dir}; best epoch {best_epoch}; counts {config["row_counts"]}')
    except (ValueError,OSError,KeyError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
