# Context transformer contract

The contextual transformer is an additional presence-only model. It does not
replace or modify `train_hurdle.py`, `train_model.py`, their checkpoints, or any
existing result directory. One invocation reads one single-resolution dataset,
its excluded-bin sidecar, and the existing genomic split, then writes a new
`context_transformer_*` result directory.

## Examples and labels

Every measured dataset row is a potential center example. The binary target is
one when its raw `break_count` is positive and zero otherwise; genuine zeros are
retained. A centered window contains an odd number of zero-anchored consecutive
bins. Its complete span must fit in the center's train, validation, or test
partition after the requested boundary buffer. Windows never cross chromosomes
or partitions. Centers lacking a complete coordinate grid or partition-safe
context are unavailable.

The sidecar supplies genomic positions and partially observed features inside a
window. A finite OK-seq or GRO-seq value is standardized; a missing value is
replaced by the training mean after transformation (standardized value zero) and
accompanied by a zero measurement mask. Sidecar rows are context only and never
become labeled center examples. Scaling is fit from measured training-center
rows only.

Each token has six features in this order: standardized OK-seq, standardized
`log1p(GRO-seq)`, OK-seq-present mask, GRO-seq-present mask, relative position
from -1 to 1, and center indicator.

## Architecture and optimization

The default model projects six token features to 64 dimensions, applies three
batch-first transformer encoder layers with four attention heads, 128-unit
feed-forward sublayers, GELU activation, pre-normalization, and dropout 0.2.
The center token and mean-pooled sequence representations are concatenated and
passed through `128 -> 64 -> 1` classification layers. The only model target is
presence of one or more observed junctions in the center bin.

Balanced training may weight positives by training negatives/positives.
Reported probabilities subtract the corresponding log weight from logits.
Unweighted validation binary log loss selects the checkpoint. Test labels do
not influence fitting, preprocessing, checkpoint selection, window size, or
other hyperparameters.

## Outputs

The new output directory contains `config.json`, `model.pt`, `history.tsv`,
`metrics.tsv`, `calibration.tsv`, `predictions.tsv`,
`break_probability.bedGraph`, and `unavailable.bed`. Metrics include prevalence,
average precision, Brier score, log loss, and precision/recall at the top 1%,
5%, and 10%, compared with constant training prevalence. Configuration records
input/code hashes, versions, seeds, architecture, feature order, scaler, class
weight, eligible counts, and completion status. Existing paths are refused.
The training command uses CUDA automatically when available and records the
resolved device; `--device cpu` and `--device cuda` provide explicit control.

The initial 10-kb experiment is:

```bash
python scripts/train_context_transformer.py \
  --dataset datasets/missing_aware_v1/bins_10kb/dataset.tsv \
  --excluded-bins datasets/missing_aware_v1/bins_10kb/excluded_bins.tsv \
  --split-dir splits/seed_20261009 \
  --window-bins 201 \
  --boundary-buffer-bp 30000 \
  --model-seed 42 --validation-seed 123
```

The 201-bin span itself enforces approximately 1 Mb of context on each side;
the additional 30-kb buffer protects the effective support of the smoothed
input signal at partition transitions.

## Required tests

Tests cover model tensor shape, retained zero labels, missing-value masks,
train-only scaling, consecutive centered coordinates, chromosome and partition
boundaries, incomplete grids, deterministic configuration, output refusal, and
a tiny end-to-end run.
