# U2OS ML break-site prediction

## Project goal

This project tests whether replication and transcription measurements can predict where DNA breaks are observed in U2OS cells after aphidicolin treatment. The model operates on fixed genomic bins and predicts the observed HTGTS break count for each bin.

The intended inputs are OK-seq replication-fork-directionality signal, GRO-seq transcription signal, and genomic bin size. The initial prediction resolutions are 5 kb and 10 kb. The same genomic split is reused at both resolutions so changing bin size does not change the biological regions used for training and testing.

The HTGTS `.tlx` files are experimental observations used to make target labels. They are not model inputs. A successful model must predict the held-out break distribution better than a training-derived baseline.

## Project status

Completed:

- Organized the project into raw inputs, references, shared datasets, splits, scripts, and model results.
- Identified the U2OS APH break-site files for chromosomes 2, 5, and 11.
- Added U2OS, IMR90, K562, and HeLa GRO-seq BAM links and their indexes.
- Replaced the low-count U2OS AT1 link with the higher-count `AT1_AltN259_full.sorted.bam` source.
- Identified 34 RDC genes whose coordinates match between the two supplied coordinate lists. Six genes with conflicting coordinates are excluded from the matched RDC list.
- Generated a reproducible 5 Mb-block genomic split with 28 RDC genes in the original training regions and 6 in testing regions.
- Implemented the shared-TSV loader and neural-network training CLI.
- Tested the loader and model with synthetic tables; all eight contract tests pass.

Status update following the data-preparation review:

- Shared 5 kb and 10 kb TSVs are now present locally. They were produced by the
  original builder and remain unchanged while its preprocessing assumptions
  are reviewed.
- The dataset builder and read-only TLX auditor now have dedicated Alt Lab
  coding-agent specifications and contract tests. See
  `docs/dataset_builder_spec.md` and `docs/tlx_audit_spec.md` for their exact
  behavior and scientific choices that still require provenance.

Not completed:

- The local `data/raw/breaksites/` and `data/raw/okseq/` directories are empty;
  their source files are on RDT01154 according to the user's listing. Production
  audits and any revised dataset build must run where those files are available.
- The `.tlx` schema, filtering rules, and normalization must be inspected before finalizing target labels.
- No model has been trained or evaluated on real sequencing data.

## Repository layout

```text
data/raw/breaksites/                 Original HTGTS files
data/raw/groseq/                     BAM files and indexes
data/raw/okseq/                      OK-seq BigWig files
data/reference/                      Gene lists and chromosome sizes
datasets/bins_5kb/dataset.tsv        Shared 5 kb feature/label table
datasets/bins_10kb/dataset.tsv       Shared 10 kb feature/label table
splits/seed_<number>/                Region-only random splits
scripts/ml_data.py                   Shared TSV loading and bin selection
scripts/train_model.py               Model definition and training CLI
results/run_<UTC>_<id>/              Model-specific outputs
```

Feature and label rows are stored once per bin size. A split stores only BED regions and metadata; it does not copy those rows for every random seed.

## Reference inputs

The corrected U2OS AT1 BAM is aligned to hg19 according to its BAM header. Chromosome lengths were extracted from that header for chr1–22, chrX, and chrY. Mitochondrial DNA is not included in the current split.

The active RDC reference is `data/reference/u2os_rdc_genes_matched34.txt`. Coordinates are currently assumed to be zero-based, start-inclusive, and end-exclusive. This assumption has not been independently verified against the annotation source and is recorded in the split manifest.

## Genomic split

The split generator uses 5 Mb genomic blocks and randomizes block assignments within every chromosome. Blocks crossed by a matched RDC gene are joined so the gene remains entirely in one partition. The target is approximately 80% of each chromosome for training and 20% for testing, with at least 17 matched RDC genes in training and at least one in testing.

The existing split is:

```text
splits/seed_20261009/train_regions.bed
splits/seed_20261009/test_regions.bed
splits/seed_20261009/gene_assignments.tsv
splits/seed_20261009/manifest.json
```

The split covers each listed nuclear chromosome exactly once and has no overlap between training and testing regions. The seed is recorded, so another split can be reproduced exactly. The split itself has no mappability mask or boundary buffer.

## Shared dataset contract

Each shared TSV must contain one row per full genomic bin and these columns:

```text
bin_id  chrom  start  end  bin_size_bp  okseq  groseq  break_count
```

`okseq` is the bin-level RFD summary and must be in the expected RFD range.
`groseq` is nonnegative aligned-read depth averaged over the bin, without
library-size normalization. `break_count` counts retained HTGTS records under
an explicit coordinate and filtering policy. The revised builder excludes
entirely missing OK-seq bins and requires a choice for partially covered bins.
The original builder replaced missing OK-seq with zero; that behavior is a
limitation of the already-generated tables, not the revised contract.

The HTGTS files must be inspected before constructing `break_count`. In particular, the chr5 filename indicates normalization, so normalized values must not be combined with raw counts without an explicit conversion or a different target definition.

## Neural network

The implemented network has three inputs, five hidden layers, and one output:

```text
3 → 64 → 64 → 32 → 16 → 8 → 1
```

The inputs are OK-seq signal, GRO-seq signal, and bin size. Hidden layers use ReLU. The output uses Softplus plus a small positive floor, producing a positive expected break count.

GRO-seq is transformed with `log1p`; bin size is transformed with the natural log. All three transformed inputs are standardized using training rows only. Training uses Adam and Poisson negative log-likelihood for raw count targets. Early stopping selects the best epoch using a validation subset drawn from whole original training-region groups. Testing remains reserved for final evaluation.

Metrics include mean absolute error, root mean squared error, and Poisson deviance. They are reported overall and separately for each bin size, alongside a baseline derived only from training break rate and genomic exposure.

## Boundary handling

The loader assigns a bin only when its full interval fits within one partition. Bins crossing train/validation/test boundaries are excluded. A configurable buffer can trim both sides of partition transitions; chromosome ends are not trimmed. The supplied OK-seq filenames indicate 30 kb smoothing on each side, so 30 kb is a reasonable initial buffer for a model run, but the effective feature window must be confirmed during feature construction.

## Reproducible model runs

Run the model only after creating shared TSVs:

```bash
python3 scripts/train_model.py \
  --dataset datasets/bins_5kb/dataset.tsv \
  --dataset datasets/bins_10kb/dataset.tsv \
  --split-dir splits/seed_20261009 \
  --target-kind raw-count \
  --boundary-buffer-bp 30000 \
  --model-seed 42 \
  --validation-seed 123
```

Each run records dataset hashes, split hashes, seeds, feature order, preprocessing statistics, architecture, software versions, and code hashes. The checkpoint stores network weights and training-derived preprocessing parameters. Input datasets are not copied into the result directory.

## Interpretation limits

The model predicts observed HTGTS signal under the selected assay and processing definitions. A high prediction does not by itself prove that a region is intrinsically break-prone, and a zero observed count is not proof that no break occurred. Performance must be evaluated on held-out genomic regions, with transcription-only, replication-only, and combined-feature baselines once the feature table is available.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

These tests use synthetic tables to check file contracts, region isolation, train-only preprocessing, architecture, checkpoint reload, reproducibility, and the absence of duplicated input tables. They do not measure biological accuracy.
