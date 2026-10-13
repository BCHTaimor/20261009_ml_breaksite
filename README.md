# U2OS break-site prediction

Predict observed HTGTS breaks using OK-seq replication and GRO-seq transcription
measurements. Shared TSV datasets are built once per prediction-bin size; random
splits store only regions and metadata. Model runs reuse the datasets without
copying training or testing tables.

For a walkthrough of the three inputs, 5- and 30-hidden-layer models, and the
files produced by training, see [Model inputs and outputs](docs/model_inputs_and_outputs.md).

For the missing-data revision and two-stage presence/count baseline, use the
[server workflow and commands](docs/hurdle_workflow.md). This preserves original
datasets and leaves full rebuilds and training to the researcher.

## Layout

```text
data/
  raw/breaksites/                Original HTGTS TLX files
  raw/groseq/                    Original BAMs and indexes
  raw/okseq/                     Original BigWig files
  reference/                    Gene lists and chromosome lengths
datasets/
  bins_5kb/                     Shared dataset.tsv
  bins_10kb/                    Shared dataset.tsv
  build_manifest.json           New builds: provenance, policies, completion status
splits/
  seed_<number>/
    manifest.json               Seed, input hashes, settings, coordinate assumptions
    train_regions.bed           Training regions, zero-based and half-open
    test_regions.bed            Testing regions, zero-based and half-open
    gene_assignments.tsv        Coordinates and partition of each RDC gene
scripts/
  generate_genomic_split.py      Working split generator
  audit_tlx.py                   Read-only TLX record and flag audit
  build_shared_datasets.py       In-place BAM/BigWig/TLX summarization
  ml_data.py                     Shared TSV loading and genomic selection
  train_model.py                 Three-input count network and training CLI
results/
  run_<UTC-datetime>_<id>/       Future model run outputs, no input copies
```

Raw files and references have been moved into `data/`, not copied. The corrected
AT1 input is `data/raw/groseq/U2OS_AT1.full.sorted.bam` (original source basename
`AT1_AltN259_full.sorted.bam`). The chromosome sizes were extracted from that
BAM's header, which identifies the hg19 alignment reference. They include
chr1–22, chrX, and chrY; mitochondrial DNA is excluded.

The 34 matched RDC gene coordinates are **assumed** zero-based, start-inclusive,
end-exclusive, as authorized by the user; this is not independently verified.
The other six genes are not counted as negative RDC examples merely because they
are absent from the matched list. No separate genomic exclusion mask exists yet.

## Generate a split

Requires Python 3.9+; no third-party packages. Run from the project root:

```bash
python3 scripts/generate_genomic_split.py \
  --chrom-sizes data/reference/u2os_at1_hg19_nuclear.chrom.sizes \
  --genome-build hg19 \
  --gene-coordinates bed \
  --coordinate-note "Gene coordinates assumed zero-based half-open; not verified"
```

Omitting `--seed` generates a random 63-bit integer and saves it in the manifest.
The output directory defaults to `splits/seed_<number>/`. To reproduce a split,
provide its saved seed and identical inputs/settings. Existing directories are
never overwritten; use `--output-dir` to reproduce into a new directory:

```bash
python3 scripts/generate_genomic_split.py \
  --chrom-sizes data/reference/u2os_at1_hg19_nuclear.chrom.sizes \
  --genome-build hg19 --gene-coordinates bed \
  --seed 20261009 --output-dir /tmp/u2os_split_reproduction
```

Defaults: 5 Mb blocks, approximately 80% training span per chromosome, at least
17 training RDC genes and at least one testing RDC gene. Blocks crossed by a
matched RDC gene are joined before randomized assignment. The existing split
has 28 training and 6 testing RDC genes. This is constrained random sampling,
not uniform sampling of every possible split.

## Shared dataset contract

Each bin-size directory will hold one `dataset.tsv` containing a header and one
row per genomic bin. The implemented raw-count model requires these columns:
`bin_id`, `chrom`, `start`, `end`, `bin_size_bp`, `okseq`, `groseq`, `break_count`.
`okseq` is mean RFD in [-1,1]; `groseq` is nonnegative mean aligned-read depth
(aligned bases divided by bin length, without library-size normalization);
`break_count` is a raw
nonnegative integer HTGTS count. These biological summaries still need to be
constructed and validated from the source files. The bin ID should include assembly and interval, e.g.
`hg19:chr1:0-5000`. Missing measurements must not be silently replaced with zero.

New builds write `build_manifest.json` at their output root. It records assembly,
bin sizes, coordinate/filtering/missing-signal policies, counts, timestamps,
completion status, and hashes for TLX/reference/code/output files. BAM/BigWig
paths, sizes and modification times are recorded without an extra full-file
hash pass; these metadata are not a cryptographic identity.
Once used by a model run, datasets must not be silently overwritten. Changed
processing or sources require a separate dataset version. No per-seed dataset
copies or persisted bin-assignment tables are needed.

## Runtime selection contract

The implemented model loader:

1. Read the chosen shared dataset and split manifest.
2. Merge adjacent regions with the same partition in memory.
3. Select a bin only if its full half-open interval fits in a partition's allowed
   region. Exclude bins that cross a training/testing boundary.
4. Apply an explicitly recorded boundary buffer at train/test transitions;
   its width is still to be decided using signal smoothing and feature windows.
   A null buffer in the split manifest means unresolved, not zero.
5. Fit learned transformations using training rows only.

Bin grids are anchored to chromosome position zero and reused across seeds.
Partial chromosome-end bins are rejected: construct only full-sized bins. Split BEDs
currently cover whole nuclear chromosomes; no mappability masks or boundary
buffers have been applied. Region membership and bin exclusions are computed
in memory, not written as duplicate training/testing datasets.

## Model runs

Each `results/run_<UTC-datetime>_<id>/` directory contains:

- `config.json`: dataset path/hash, split path and region hashes, split seed,
  model seed, feature/target columns, boundary buffer, preprocessing and model
  settings, and code version/hash.
- `model/`: trained model and any fitted transformations.
- `metrics.tsv`: evaluation results.
- `history.tsv`: training and validation loss by epoch.
- `validation_regions.bed`: small region list held out from original training
  regions for early stopping. The source split stays unchanged.
- `predictions.tsv`: optional, only when needed for downstream analysis.

Split seeds control genomic membership; model seeds control model fitting.
Multiple model seeds on the same split reuse all input data.

## Current status and checks

The split generator, data loader, and neural-network training CLI are ready.
Existing 5 kb and 10 kb tables were generated with the initial builder; they
are preserved as-is and have not been scientifically validated. The original
builder assumed one-based TLX junctions, counted readable records without flag
filtering, and replaced missing OK-seq signal with zero. Those choices require
review before interpreting a model trained on those tables.

The large TLX and OK-seq source files are present on RDT01154 according to the
user's directory listing, but are not visible in this local workspace. File
audits and production rebuilding therefore run on RDT01154. Tests here use
small purpose-built fixtures and cannot establish filtering or normalization
provenance of the real inputs.

## Alt Lab coding workflow

Project code changes follow the user's requested Alt Lab coding-agent workflow:
write the public behavior specification, derive a test matrix from its contract,
implement, and verify with those tests. The dataset builder and TLX auditor have
dedicated specifications and test files. Tests of file handling and numerical
summaries do not determine the scientifically appropriate TLX filtering policy.

- `docs/dataset_builder_spec.md`: feature definitions, coordinate/filtering
  choices, missing-signal handling, output protection, and provenance.
- `docs/tlx_audit_spec.md`: literal flag counts and read-only record auditing.
- `tests/test_dataset_builder.py` and `tests/test_tlx_audit.py`: contract tests.

The first production step is the non-mutating audit:

```bash
python3 scripts/audit_tlx.py data/raw/breaksites/*.tlx
```

Use the builder specification and `python3 scripts/build_shared_datasets.py
--help` for the explicit choices required before rebuilding. Do not infer
coordinate origin from BAM conventions or filtering rules from flag names.
Write a revised dataset to a new version directory; never overwrite the
existing scientific tables. Review the chr5 normalization provenance before
deciding how to combine the three TLX sources.

The layout migration was checked by regenerating seed 20261009 and comparing
the BEDs and gene assignments byte-for-byte. Random-seed recording and refusal
to overwrite an existing split were also checked. The generator validates
disjoint full chromosome coverage and intact RDC genes before writing outputs.

## Neural network

Architecture: **3 → 64 → 64 → 32 → 16 → 8 → 1**, with ReLU on five hidden
layers and Softplus plus a small positive floor on the output. The three inputs
are OK-seq RFD, GRO-seq signal, and bin size. Output is an expected observed
break count, not a probability or proof of intrinsic break susceptibility.

GRO-seq is transformed with log1p; bin size with natural log. All three inputs
are standardized using fitting rows only. Training uses Adam, Poisson count
loss, and validation-based early stopping. Validation is sampled by whole
original training region groups, keeping RDC genes intact. Test labels are
used for metrics only after the best epoch is selected. Metrics include MAE,
RMSE, and Poisson deviance, overall and per bin size, compared with a constant
break-rate baseline fitted only on training counts and genomic exposure.

Multiple datasets can be supplied together; each row has equal loss weight,
so finer resolutions contribute more rows. Overlapping 5/10 kb examples are
correlated; their shared genomic partition prevents cross-set overlap but does
not make them independent. A bin size outside those trained on is unvalidated.

Install dependencies into your analysis environment:

```bash
python3 -m pip install -r requirements.txt
```

After building the real TSVs, an example joint-resolution run is:

```bash
python3 scripts/train_model.py \
  --dataset datasets/bins_5kb/dataset.tsv \
  --dataset datasets/bins_10kb/dataset.tsv \
  --split-dir splits/seed_20261009 \
  --target-kind raw-count \
  --boundary-buffer-bp 30000 \
  --model-seed 42 --validation-seed 123
```

The 30 kb buffer is an explicit starting value based on the supplied OK-seq
smoothing filenames (30 kb each side); verify effective signal support and
increase it if feature construction uses wider windows. Buffers apply on both
sides of train/validation/test transitions, not chromosome ends. The split's
17-gene training minimum refers to the original training pool, before holding
out validation or excluding edge bins.

Raw-count mode deliberately rejects fractional/negative targets. An integer
column alone does not establish that labels are raw: inspect the normalized
chr5 TLX file before constructing or combining labels. All supplied datasets
must use the same verified assembly, assay definitions, and normalization.

The command refuses existing output directories. Omit `--output-dir` for a
unique UTC datetime/random-suffix run directory. Add `--save-predictions` only
when those extra outputs are needed; no input dataset copies are made.

The checkpoint `model/model.pt` stores `state_dict`, architecture, and fitted
preprocessor mean/scale. Reload with `torch.load(..., map_location='cpu',
weights_only=True)`, construct `BreakCountNet`, and load its `state_dict`.
Apply `Preprocessor.raw_transform` followed by the saved mean/scale before
prediction. The run configuration records input/code hashes and versions.

## Context transformer

`scripts/train_context_transformer.py` is a separate presence-only model; it
does not replace the pointwise hurdle or count models. For each measured center
bin, it reads an odd window of consecutive bins and predicts whether the center
has at least one observed HTGTS junction. Genuine zero-count centers remain
training examples. Excluded-bin rows provide partially observed context but do
not become labeled centers.

The default 10 kb run uses 201 tokens, spanning 2.01 Mb. Each token contains
scaled OK-seq, scaled log1p GRO-seq, separate measurement-present masks,
relative position, and a center indicator. Six features are projected to width
64 and passed through three transformer encoder layers with four attention
heads and 128-unit feed-forward sublayers. The center and mean-pooled sequence
representations feed a binary classifier. Complete windows must remain within
one chromosome and one train/validation/test partition.

Run the initial experiment on RDT01154:

```bash
python scripts/train_context_transformer.py \
  --dataset datasets/missing_aware_v1/bins_10kb/dataset.tsv \
  --excluded-bins datasets/missing_aware_v1/bins_10kb/excluded_bins.tsv \
  --split-dir splits/seed_20261009 \
  --window-bins 201 \
  --boundary-buffer-bp 30000 \
  --model-seed 42 --validation-seed 123
```

The completed seed-42 run had 5,466 training, 782 validation, and 1,134 test
windows. On the 1,134-bin test subset shared with the pointwise hurdle model,
the transformer achieved average precision 0.151 versus 0.135 for the hurdle
model and prevalence 0.092. Its top 5% contained 13 positives among 57 bins
(22.8% precision), compared with 9 among 57 (15.8%) for the pointwise model.
Brier score improved from 0.0830 to 0.0827 and log loss from 0.3063 to 0.3040.

This is evidence that surrounding genomic context adds modest ranking signal,
not that the model identifies a causal break mechanism. The comparison uses one
genomic split and one model seed. OK-seq is sparse within the windows (about 13%
mean and 8% median observed-bin fraction), so GRO-seq context and measurement
masks may account for some of the gain. Replicate seeds and GRO-only, OK-only,
and mask-only ablations are required before biological interpretation. See
`docs/context_transformer_spec.md` for the complete contract and outputs.

Run contract and synthetic end-to-end tests:

```bash
python3 -m unittest discover -s tests -v
```

The original synthetic suite was validated with Python 3.9, NumPy 2.0.2, and
PyTorch 2.8.0 on CPU. The contextual transformer completed on the real 10 kb
dataset with Python 3.11.0, NumPy 2.4.3, PyTorch 2.14.1+cu130, and scikit-learn
1.9.1 on CPU. Synthetic test performance is not evidence of biological
predictive accuracy; real-run metrics are descriptive of the recorded split.

API references: [PyTorch PoissonNLLLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html)
and [Softplus](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html).
