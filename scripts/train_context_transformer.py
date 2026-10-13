#!/usr/bin/env python3
"""Train a leakage-safe contextual transformer for break presence."""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from ml_data import classify_bin, load_datasets, load_regions


FEATURE_ORDER = [
    'scaled_okseq', 'scaled_log1p_groseq', 'okseq_present',
    'groseq_present', 'relative_position', 'center_indicator',
]


class ContextTransformer(nn.Module):
    """Small encoder that predicts presence in the center genomic bin."""

    def __init__(self, n_features=6, width=64, layers=3, heads=4,
                 feedforward=128, dropout=.2):
        super().__init__()
        self.embedding = nn.Linear(n_features, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def forward(self, x):
        """Return one logit per BxLx6 input window without changing it."""
        encoded = self.encoder(self.embedding(x))
        center = encoded[:, encoded.shape[1] // 2]
        pooled = encoded.mean(dim=1)
        return self.classifier(torch.cat((center, pooled), dim=1)).squeeze(-1)


def corrected_probability(logits, pos_weight):
    """Undo the analytical log-odds shift caused by positive weighting."""
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError('Positive class weight must be finite and positive')
    return torch.sigmoid(logits - math.log(pos_weight))


def fit_scaler(x):
    """Fit OK and log1p-GRO mean/scale from finite measured training rows."""
    x = np.asarray(x, dtype=float)
    if (x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x[:, :2]).all()
            or np.any(x[:, 1] < 0) or not len(x)):
        raise ValueError('Invalid training features')
    transformed = np.column_stack((x[:, 0], np.log1p(x[:, 1])))
    mean, scale = transformed.mean(axis=0), transformed.std(axis=0)
    scale[scale < 1e-12] = 1.
    return mean, scale


def read_context_grid(dataset_rows, excluded_path, bin_size):
    """Return coordinate-keyed raw context features from measured and sidecar rows."""
    grid = {}
    for row in dataset_rows:
        key = (row['chrom'], int(row['start']))
        grid[key] = (float(row['okseq']), float(row['groseq']))
    if excluded_path is not None:
        with Path(excluded_path).open() as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            required = {'bin_id', 'chrom', 'start', 'end', 'bin_size_bp',
                        'okseq', 'groseq', 'exclusion_reason'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Excluded TSV lacks required columns')
            for line, row in enumerate(reader, 2):
                try:
                    start, end, size = (int(row[k]) for k in
                                        ('start', 'end', 'bin_size_bp'))
                    if size != bin_size or start < 0 or end - start != size or start % size:
                        raise ValueError('invalid coordinate or resolution')
                    key = (row['chrom'], start)
                    if key in grid:
                        raise ValueError('duplicate coordinate')
                    values = []
                    for name in ('okseq', 'groseq'):
                        value = float(row[name])
                        values.append(value if math.isfinite(value) else float('nan'))
                    if math.isfinite(values[0]) and not -1 <= values[0] <= 1:
                        raise ValueError('OK-seq outside [-1,1]')
                    if math.isfinite(values[1]) and values[1] < 0:
                        raise ValueError('negative GRO-seq')
                    grid[key] = tuple(values)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{excluded_path}:{line}: {exc}') from exc
    return grid


def build_windows(data, grid, regions, window_bins, buffer_bp, mean, scale):
    """Build centered windows whose complete spans fit one genomic partition."""
    if window_bins < 1 or window_bins % 2 != 1:
        raise ValueError('window_bins must be a positive odd integer')
    if buffer_bp < 0:
        raise ValueError('boundary buffer must be nonnegative')
    sizes = {int(row['bin_size_bp']) for row in data['rows']}
    if len(sizes) != 1:
        raise ValueError('Use one bin resolution per invocation')
    size = next(iter(sizes))
    mean, scale = np.asarray(mean), np.asarray(scale)
    if mean.shape != (2,) or scale.shape != (2,) or np.any(scale <= 0):
        raise ValueError('Invalid scaler')
    radius = window_bins // 2
    relative = np.linspace(-1., 1., window_bins, dtype=np.float32)
    windows, labels, indices, partitions = [], [], [], []
    by_chrom = {}
    for region in regions:
        by_chrom.setdefault(region[0], []).append(region)
    for index, row in enumerate(data['rows']):
        chrom, start = row['chrom'], int(row['start'])
        span_start, span_end = start - radius * size, start + (radius + 1) * size
        if span_start < 0:
            continue
        part = classify_bin(chrom, span_start, span_end,
                            by_chrom.get(chrom, []), buffer_bp)
        if part == 'excluded' or part != data['partition'][index]:
            continue
        tokens = np.empty((window_bins, 6), dtype=np.float32)
        complete = True
        for j, position in enumerate(range(span_start, span_end, size)):
            raw = grid.get((chrom, position))
            if raw is None:
                complete = False
                break
            ok, gro = raw
            ok_present, gro_present = math.isfinite(ok), math.isfinite(gro)
            tokens[j] = (
                (ok - mean[0]) / scale[0] if ok_present else 0.,
                (math.log1p(gro) - mean[1]) / scale[1] if gro_present else 0.,
                float(ok_present), float(gro_present), relative[j],
                float(j == radius),
            )
        if complete:
            windows.append(tokens)
            labels.append(float(data['y'][index] > 0))
            indices.append(index)
            partitions.append(part)
    shape = (0, window_bins, 6) if not windows else None
    array = np.empty(shape, dtype=np.float32) if shape else np.stack(windows)
    return {'x': array, 'y': np.asarray(labels, dtype=np.float32),
            'row_index': np.asarray(indices, dtype=int),
            'partition': np.asarray(partitions)}


def evaluate(y, probability):
    """Return presence ranking and calibration metrics on natural prevalence."""
    y, probability = np.asarray(y, dtype=float), np.asarray(probability, dtype=float)
    if (y.ndim != 1 or probability.shape != y.shape or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(probability).all()
            or np.any((y != 0) & (y != 1))
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Invalid metric vectors')
    q = np.clip(probability, 1e-12, 1 - 1e-12)
    result = {
        'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
        'average_precision': float(average_precision_score(y, probability))
        if y.any() and not y.all() else float('nan'),
        'brier': float(np.mean((probability - y) ** 2)),
        'log_loss': float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))),
    }
    order = np.argsort(-probability, kind='stable')
    for fraction in (.01, .05, .1):
        k = max(1, math.ceil(len(y) * fraction))
        found = int(y[order[:k]].sum())
        suffix = f'{int(fraction * 100)}pct'
        result[f'precision_top_{suffix}'] = found / k
        result[f'recall_top_{suffix}'] = found / int(y.sum()) if y.any() else float('nan')
    return result


def write_tsv(path, rows, fields=None):
    """Create one TSV and refuse overwrite."""
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    """Return a file SHA256 without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fit_model(model, x_train, y_train, x_val, y_val, args, weight):
    """Fit a new model and restore the validation-selected checkpoint."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    best, stale, state, history = float('inf'), 0, None, []
    weight_tensor = torch.tensor(weight, dtype=torch.float32, device=x_train.device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for batch in torch.randperm(len(y_train)).split(args.batch_size):
            logits = model(x_train[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_train[batch], pos_weight=weight_tensor)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            logits = model(x_val) - math.log(weight)
            validation_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_val).item()
        history.append({'epoch': epoch, 'train_loss': total / len(y_train),
                        'validation_loss': validation_loss})
        print(f'epoch {epoch}: train={total/len(y_train):.6g} '
              f'validation={validation_loss:.6g}', flush=True)
        if validation_loss < best:
            best, stale, state = validation_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(state)
    return history


def main(argv=None):
    """Train one contextual transformer and create a new results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--excluded-bins', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--window-bins', type=int, default=201)
    parser.add_argument('--boundary-buffer-bp', type=int, default=30000)
    parser.add_argument('--validation-fraction', type=float, default=.15)
    parser.add_argument('--validation-seed', type=int, default=123)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--class-weight', choices=('balanced', 'none'), default='balanced')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if (args.window_bins < 1 or args.window_bins % 2 != 1
            or min(args.width, args.layers, args.heads, args.feedforward,
                   args.epochs, args.patience, args.batch_size) < 1
            or args.width % args.heads or not 0 <= args.dropout < 1
            or args.boundary_buffer_bp < 0
            or args.learning_rate <= 0 or args.weight_decay < 0):
        raise ValueError('Invalid architecture, window, or optimization settings')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
        if args.device != 'auto' else 'cpu')

    regions = load_regions(args.split_dir, args.validation_fraction,
                           args.validation_seed)
    data = load_datasets([args.dataset], regions, args.boundary_buffer_bp)
    sizes = {int(row['bin_size_bp']) for row in data['rows']}
    if len(sizes) != 1:
        raise ValueError('Use one bin resolution per invocation')
    bin_size = next(iter(sizes))
    train_rows = data['x'][data['partition'] == 'train']
    mean, scale = fit_scaler(train_rows)
    grid = read_context_grid(data['rows'], args.excluded_bins, bin_size)
    windows = build_windows(data, grid, regions, args.window_bins,
                            args.boundary_buffer_bp, mean, scale)
    masks = {part: windows['partition'] == part
             for part in ('train', 'validation', 'test')}
    for part in ('train', 'validation'):
        labels = windows['y'][masks[part]]
        if not len(labels) or not labels.any() or labels.all():
            raise ValueError(f'{part} must contain positive and negative windows')
    if not masks['test'].any():
        raise ValueError('No eligible test windows')
    positives = int(windows['y'][masks['train']].sum())
    weight = ((int(masks['train'].sum()) - positives) / positives
              if args.class_weight == 'balanced' else 1.)

    output = args.output_dir or Path('results') / (
        'context_transformer_' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' +
        uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    code_path = Path(__file__)
    inputs = [args.dataset, args.excluded_bins,
              args.split_dir / 'train_regions.bed',
              args.split_dir / 'test_regions.bed']
    config = {
        'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'inputs_sha256': {str(path.resolve()): sha256(path) for path in inputs},
        'code_sha256': {code_path.name: sha256(code_path),
                        'ml_data.py': sha256(code_path.parent / 'ml_data.py')},
        'versions': {'python': sys.version, 'torch': torch.__version__,
                     'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'resolved_device': str(device),
        'architecture': {'type': 'context_transformer', 'width': args.width,
                         'layers': args.layers, 'heads': args.heads,
                         'feedforward': args.feedforward, 'dropout': args.dropout},
        'feature_order': FEATURE_ORDER,
        'scaler': {'mean': mean.tolist(), 'scale': scale.tolist()},
        'positive_weight': weight,
        'probability_correction': 'subtract log positive weight; calibration not guaranteed',
        'eligible_windows': {part: int(mask.sum()) for part, mask in masks.items()},
        'validation_regions': regions,
    }
    with (output / 'config.json').open('x') as handle:
        json.dump(config, handle, indent=2)

    torch.manual_seed(args.model_seed)
    torch.use_deterministic_algorithms(True)
    x = torch.tensor(windows['x'], dtype=torch.float32, device=device)
    y = torch.tensor(windows['y'], dtype=torch.float32, device=device)
    indices = {part: torch.tensor(np.flatnonzero(mask), dtype=torch.long,
                                  device=device)
               for part, mask in masks.items()}
    model = ContextTransformer(6, args.width, args.layers, args.heads,
                               args.feedforward, args.dropout).to(device)
    history = fit_model(model, x[indices['train']], y[indices['train']],
                        x[indices['validation']], y[indices['validation']],
                        args, weight)
    write_tsv(output / 'history.tsv', history)
    model.eval()
    probabilities = np.empty(len(y), dtype=float)
    with torch.no_grad():
        for batch in torch.arange(len(y), device=device).split(args.batch_size):
            probabilities[batch.cpu().numpy()] = corrected_probability(
                model(x[batch]), weight).cpu().numpy()

    baseline = float(windows['y'][masks['train']].mean())
    metrics, calibration = [], []
    for part, mask in masks.items():
        for name, probability in (
                ('context_transformer', probabilities[mask]),
                ('constant_prevalence', np.full(mask.sum(), baseline))):
            metrics.append({'partition': part, 'model': name,
                            **evaluate(windows['y'][mask], probability)})
            bucket = np.minimum((probability * 10).astype(int), 9)
            for index in range(10):
                selected = bucket == index
                calibration.append({
                    'partition': part, 'model': name,
                    'probability_low': index / 10,
                    'probability_high': (index + 1) / 10,
                    'n': int(selected.sum()),
                    'mean_probability': float(probability[selected].mean())
                    if selected.any() else float('nan'),
                    'observed_fraction': float(windows['y'][mask][selected].mean())
                    if selected.any() else float('nan'),
                })
    write_tsv(output / 'metrics.tsv', metrics)
    write_tsv(output / 'calibration.tsv', calibration)

    probability_by_row = {int(row_index): float(probabilities[i])
                          for i, row_index in enumerate(windows['row_index'])}
    predictions = []
    for index, row in enumerate(data['rows']):
        available = index in probability_by_row
        predictions.append({key: row[key] for key in ('bin_id', 'chrom', 'start', 'end')} | {
            'partition': data['partition'][index],
            'break_probability': probability_by_row.get(index, float('nan')),
            'availability': 'available' if available else 'unavailable',
            'exclusion_reason': '' if available else 'missing_or_partition_unsafe_context',
        })
    predictions.sort(key=lambda row: (row['chrom'], int(row['start'])))
    write_tsv(output / 'predictions.tsv', predictions)
    with (output / 'unavailable.bed').open('x') as handle:
        for row in predictions:
            if row['availability'] == 'unavailable':
                handle.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['exclusion_reason']}\n")
    with (output / 'break_probability.bedGraph').open('x') as handle:
        for row in predictions:
            if row['availability'] == 'available':
                handle.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['break_probability']:.9g}\n")
    torch.save({'model': model.state_dict(), 'mean': mean.tolist(),
                'scale': scale.tolist(), 'positive_weight': weight,
                'bin_size_bp': bin_size, 'window_bins': args.window_bins,
                'architecture': config['architecture']}, output / 'model.pt')
    config.update({
        'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'best_epoch': min(history, key=lambda row: row['validation_loss'])['epoch'],
    })
    with (output / 'config.json').open('w') as handle:
        json.dump(config, handle, indent=2)
    print(f'Completed: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
