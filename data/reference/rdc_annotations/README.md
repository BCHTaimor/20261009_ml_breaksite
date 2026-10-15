# U2OS RDC annotation sources

These files are immutable copies of annotation files supplied to Taimor
Williams and placed in the project on 2026-10-15. They are reference
annotations, not raw sequencing data.

## Files

- `Correct_file_of_RDC_and_RDC_candidates_together_10_7_26.update_len_strand.20261007.tsv`
  is the supplied tabular list of 260 RDC, RDC-candidate, and nonRDC regions.
  According to the accompanying email from Adam, Aseda adjusted the boundaries
  of some partially expressed genes to the observed transcription unit and two
  intergenic RDC regions were restored after comparison with a prior 48-region
  list.
- `Correct_file_of_RDC_and_RDC_candidates_together_10_7_26.update_len_strand.20261007.bed`
  is the supplied BED representation of the same 260 rows. Its start coordinate
  is one base lower than the TSV start coordinate, consistent with zero-based,
  half-open BED coordinates versus one-based tabular coordinates. This
  relationship was checked for the supplied pair but the biological boundaries
  were not independently validated.
- `U2OS_48_RDC_regions.bed` is the supplied historical 48-region RDC list.

## Intended use

RDC membership, candidate status, notes, and other annotation-derived labels
must not be model inputs or training targets. They may be used for development
evaluation and, after the modeling approach is fixed, final held-out evaluation.
The historical 48-region list is retained for provenance and comparison.

Whether transcription-unit geometry from this curated collection is an
appropriate model input remains an open design question: the collection was
assembled around RDC/candidate/nonRDC annotations and is not a genome-wide,
annotation-independent catalog. Prefer a genome-wide transcription-unit source
for predictor construction.

## SHA-256 checksums

```text
35efa3e6c84667c4b5220916db43f1758ed8ff05ca98ba6d7f69cf0b14a40eb0  Correct_file_of_RDC_and_RDC_candidates_together_10_7_26.update_len_strand.20261007.bed
2fd369e3d2d436f60eb674aaa6b7998ebbd72f75dcf534afb8233aff933ca54f  Correct_file_of_RDC_and_RDC_candidates_together_10_7_26.update_len_strand.20261007.tsv
f5bc358bac0ceb153683dad69a8e7dbfadff9fc12374cd8e63680516822979bf  U2OS_48_RDC_regions.bed
```
