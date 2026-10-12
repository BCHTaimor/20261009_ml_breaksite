# Model inputs, architecture, and outputs

This model predicts the **observed HTGTS break count in each genomic bin** from
three inputs: OK-seq signal, GRO-seq signal, and bin size. The chromosome and
coordinates identify where each prediction belongs. They are not neural-network
inputs. A prediction is an expected count, so it can be fractional; it is not a
break probability or a prediction of the exact base at which a break occurs.

## From sequencing files to one example

The dataset builder summarizes the large source files in place, without changing
them. Each full 5 kb or 10 kb interval becomes one table row:

| Source | Summary for that interval | Role |
|---|---|---|
| One selected OK-seq BigWig | Mean replication fork directionality (RFD) | Input 1 |
| One selected GRO-seq BAM | Aligned bases divided by bin length: mean aligned-read depth | Input 2 |
| Genomic interval | Length in base pairs, e.g. `5000` | Input 3 |
| Supplied HTGTS TLX files | Number of retained junction rows inside the interval | Training answer (`break_count`) |

GRO-seq is **not library-size normalized**. The revised builder combines both
strands, excludes unmapped/secondary/supplementary/duplicate/QC-failed reads,
and counts overlapping mates independently. CIGAR deletion/skipped regions do
not contribute aligned bases. The model receives no strand-specific feature.

The revised builder requires explicit TLX coordinate and filtering policies.
Every retained row contributes one count, pooled across the supplied TLX files;
there is no additional cross-file deduplication or normalization. Missing OK-seq
handling is also explicit: exclude bins with any missing signal, or use a mean
over covered positions while excluding entirely missing bins. See the
[builder specification](dataset_builder_spec.md) for the complete contract.

**Illustrative values only—not an experimental result:**

```text
bin_id                 chrom  start  end    bin_size_bp  okseq  groseq  break_count
hg19:chr1:10000-15000   chr1   10000  15000  5000         0.21   4.8     2
```

The network takes `[0.21, 4.8, 5000]`. During fitting it compares its prediction
with `2`. A prediction of `1.7` means an expected 1.7 observed breaks in that
same `chr1:10000–15000` interval. The answer is not supplied as an input.

## Shared tables and genomic partitions

Each resolution has one shared `datasets/bins_<size>kb/dataset.tsv`:

| Column | Meaning |
|---|---|
| `bin_id` | Unique assembly/interval identifier, used to join predictions back to data |
| `chrom`, `start`, `end` | Zero-based, start-inclusive/end-exclusive genomic interval |
| `bin_size_bp` | Interval length; third model input |
| `okseq` | Mean RFD in `[-1, 1]`; first model input |
| `groseq` | Nonnegative mean aligned-read depth; second model input |
| `break_count` | Nonnegative integer observed junction count; target |

Bins are anchored at chromosome coordinate zero. Partial chromosome-end bins
are omitted. Features and targets are stored once per resolution, not copied
for every random split or model run.

The split folder supplies `train_regions.bed`, `test_regions.bed`,
`gene_assignments.tsv`, and `manifest.json`. The initial split randomly assigns
5 Mb blocks within each chromosome, targeting 80% training/20% testing. Blocks
crossed by a matched RDC gene are kept together. The requirement of at least
17 of 34 RDC genes applies to the original training pool.

At runtime, approximately 15% of the original training span is held out for
validation using whole region groups. A bin must fit completely inside its
assigned region after boundary exclusions. The example command below excludes
30 kb on each side of train/validation/test transitions. The same genomic
partition is used for 5 kb and 10 kb examples; finer bins do not create a new
random split. The 30 kb buffer is a starting choice based on track smoothing,
not a demonstrated guarantee that all signal dependence is removed.

| Seed | What it controls |
|---|---|
| Split seed | Original training/testing genomic regions |
| Validation seed | Whole training groups held out for early stopping |
| Model seed | Initial neural-network weights and training batch order |

## Network and fitting

The default has **five hidden layers**:

```text
3 inputs → 64 → 64 → 32 → 16 → 8 → 1 expected break count
```

`--hidden-layers 30` selects **30 hidden layers**: 27 layers with 64 neurons
each, followed by layers of 32, 16, and 8 neurons. Every hidden layer uses ReLU;
the output uses Softplus plus `1e-6`. The three inputs and one output stay the
same. More layers do not by themselves establish better predictions.

Before fitting, GRO-seq becomes `log1p(groseq)` and bin size becomes
`log(bin_size_bp)`; OK-seq is unchanged. All three columns are standardized
using means and standard deviations fitted only on training rows. Those same
statistics are used for validation and testing and saved with the model.

Training uses Adam (default learning rate `0.001`), Poisson count loss, batches
of 1,024, and up to 100 epochs. It stops after 10 epochs without improved
validation loss and restores the best validation epoch. Test labels are used
for evaluation after that selection. Execution currently uses one CPU thread.
Supplying both resolutions trains one joint model with equal weight per row;
5 kb therefore contributes more rows, and overlapping resolutions are correlated.

## Run the 30-layer comparison

From the project root on RDT01154, using the existing shared tables:

```bash
python3 scripts/train_model.py \
  --dataset datasets/bins_5kb/dataset.tsv \
  --dataset datasets/bins_10kb/dataset.tsv \
  --split-dir splits/seed_20261009 \
  --target-kind raw-count \
  --boundary-buffer-bp 30000 \
  --hidden-layers 30 \
  --model-seed 42 \
  --validation-seed 123 \
  --save-predictions
```

This keeps the split, seeds, and other settings of the earlier command while
changing depth. Use `--hidden-layers 5` for the original architecture. Existing
datasets are reused. Each run receives a new output folder; the command refuses
to overwrite an existing output directory.

## Output files

Outputs are under `results/run_<UTC-datetime>_<random-id>/`:

| File | Contents |
|---|---|
| `config.json` | Arguments/seeds, architecture, feature order, target/loss, best epoch, completed epochs, scaling statistics, partition row counts, baseline rate, input/code hashes, software versions, and selection/weighting rules |
| `model/model.pt` | Checkpoint keys: `state_dict`, `architecture`, `preprocessor` (with `mean` and `scale`) |
| `history.tsv` | `epoch`, `train_loss`, `validation_loss` |
| `validation_regions.bed` | `chrom`, `start`, `end` of held-out validation regions, without a header |
| `metrics.tsv` | `partition`, `bin_size_bp`, `model`, `n`, `mae`, `rmse`, `poisson_deviance` |
| `predictions.tsv` | Optional: `bin_id`, `partition`, `predicted_break_count` |

Predictions contain training, validation, and testing rows, excluding unusable
boundary bins. Join them to shared datasets by `bin_id` to recover coordinates
and observed `break_count`; neither truth nor input features are duplicated in
the prediction file. Restore a checkpoint with the saved architecture's depth
and scaling statistics, not an assumed five-layer network.

Metrics compare `network` against `constant_rate`, a training-only estimate of
breaks per base multiplied by bin length. MAE, RMSE, and Poisson deviance are
all lower-is-better; results are reported per partition, per bin size, and for
all supplied sizes together. Inspect validation for model choices and testing
for held-out evaluation; completion alone does not establish predictive accuracy.

## Provenance and verification

The existing original tables assumed one-based TLX coordinates, counted rows
without flag filtering, and replaced missing OK-seq signal with zero. Updating
the builder has **not** changed those existing tables. The user's exploratory
run accepts those assumptions; their correctness and chr5 normalization remain
unresolved. A rebuild uses explicit policies and a new output root, preserving
the previous data. Integer labels alone do not establish biological validity.

Implementation: [trainer](../scripts/train_model.py),
[loader](../scripts/ml_data.py), [builder](../scripts/build_shared_datasets.py).
Tests: [model/partition tests](../tests/test_ml.py),
[builder tests](../tests/test_dataset_builder.py). Run
`python3 -m unittest discover -s tests -v` for code-contract tests; fixture-based
tests do not measure real biological performance. See the [README](../README.md)
for setup and the broader repository layout.
