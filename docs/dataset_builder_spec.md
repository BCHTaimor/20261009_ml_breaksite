# Shared dataset builder contract

This specification precedes the revised implementation and tests. It does not
establish the biological filtering rules or coordinate convention of the real
TLX files. Those remain choices requiring source provenance.

## Public command

`scripts/build_shared_datasets.py` reads immutable chromosome sizes, one BAM,
one BigWig and one or more tab-delimited TLX files. It creates shared complete,
zero-anchored genomic bins for each requested bin size (default 5000, 10000).
Chromosome tails shorter than a bin are omitted. All files must agree on
chromosome lengths for selected chromosomes. No assembly conversion occurs.

Required semantic arguments: `--genome-assembly`, `--tlx-coordinate-system`
(`zero-based` or `one-based`), `--tlx-filter-policy` (`all-rows` or `zero-flags`),
`--label-policy pooled-retained-rows`, and `--missing-okseq-policy`
(`exclude-any-missing` or `covered-mean`). The latter retains partially covered
bins only with an explicit request; entirely missing bins are always excluded.
With `zero-flags`, repeat `--zero-flag COLUMN` for each flag that must numerically
equal zero. No flag meaning is inferred. No weighting or deduplication is done.
Pooling explicitly counts each retained row once, including identical records
in distinct files; it does not establish biological independence or interpret
the filename `norm`. Repeated references to the same source file are rejected.

## Dataset contract

Each `bins_5kb/dataset.tsv` (and other sizes) contains `bin_id`, `chrom`, `start`,
`end`, `bin_size_bp`, `okseq`, `groseq`, `break_count`. Bin IDs include the explicit
assembly. Coordinates are zero-based half-open. OK-seq is the exact length-
weighted mean of covered bases, never a zero replacement for missing values.
Finite mean RFD outside [-1, 1] fails rather than being silently excluded.
GRO-seq is aligned base count / bin length from primary mapped reads, excluding
QC-failed and duplicate-marked reads. M/= /X contribute; deletion and skipped
reference intervals do not. Overlapping mates count independently, both strands
are combined, no MAPQ/base-quality threshold or library normalization applies.
This is mean read depth, not RPKM. Input BAM is traversed once for all bin sizes.

TLX fields `Rname` and `Junction` are parsed by tabs, preserving empty cells.
Malformed rows, invalid selected flags, and out-of-range selected-chromosome
junctions fail with source/line context; unselected chromosomes are tallied and
excluded. Junction zero is legal only for zero-based sources. Each row maps
directly to its bin, avoiding per-base scans. Per-source retained/filtered/
outside-reference counts are recorded independently of bin omissions.

Outputs use exclusive creation and no overwrite mode. A root
`build_manifest.json` begins with status `running` and becomes `complete` only
after all TSVs close successfully; failures record `failed`. Existing outputs
or manifest stop a run before processing. Incomplete outputs are retained for
inspection; use a new output directory to retry. The manifest records command,
UTC time, settings, source paths/resolved targets/size/mtime, SHA256 for code,
chromosome sizes, TLX files and outputs, dependency versions, source counts and
bin exclusions. BAM/BigWig metadata is recorded without extra full-file hash
passes to avoid rereading huge inputs. Metadata is not a cryptographic identity.

## Public functions and test matrix

- `read_chrom_sizes`: validate unique positive chromosome sizes.
- `read_tlx`: count rows at each size; tab-empty fields, gzip, both conventions,
  first/last/boundary positions, filtered rows, multiple sources, tails,
  off-reference rows, malformed width/coordinate/flag failures.
- `groseq_depths`: primary mapped filtering; CIGAR gaps, crossing bins,
  overlapping reads and mates, empty bins, multiple sizes in one stream.
- `write_dataset`: full/missing/partial/zero/negative BigWig signal, exact
  interval mean, complete bins only, preserved output schema and exclusions.
- `main`: required semantic choices, reference compatibility, exclusive output
  preservation, manifest completion and hashes. Native BAM/BigWig integration
  fixtures are tiny synthetic files testing software behavior only.

Existing real datasets predate this contract and are not rebuilt by this work.
No successful software test establishes biological prediction accuracy.

## Explicit genomic exclusions

Optional `--exclude-bed PATH` supplies zero-based half-open intervals on the
selected reference. Blank/comment lines are ignored; extra BED columns are
allowed. Unknown chromosomes, negative/empty/reversed/out-of-reference intervals,
and malformed rows fail before output creation. Overlapping/adjacent intervals
are merged. Any overlap excludes the entire bin, at every requested resolution;
touching endpoints do not overlap. Counts are never changed to zero by masking.
Unmasked bins with genuine zero counts or zero RFD remain eligible.
No bait radius or exclusion interval is inferred from target counts.
The manifest records the mask path/hash and excluded-bin totals separately from
missing-OK-seq totals (mask exclusion takes precedence).

Test matrix: unsorted/overlapping/adjacent BED intervals, extra columns,
unknown chromosome/invalid bounds/invalid row; exact endpoint contact versus
one-base overlap; both resolutions; masked nonzero targets omitted, unmasked
zero targets retained; manifest mask identity and no-mask compatibility.
