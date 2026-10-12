# Read-only TLX audit contract

`scripts/audit_tlx.py` audits one or more original TLX paths, including symlinks,
without writing, filtering, converting coordinates, or recommending exclusions.
Plain UTF-8 TSV and `.gz` TSV are supported, with LF, CRLF, or CR newlines.
Empty cells are preserved. The first record must contain unique, nonempty column
names including `Rname` and `Junction`; optional known flag columns may be absent.
A BOM and surrounding whitespace on header names are tolerated.

`audit(path)` returns one JSON-compatible report. `rows` counts nonblank data
records, including malformed records. Wrong-width records increment
`malformed_rows` and are excluded from all field statistics; at most five source
line examples are retained. A header-only file is valid. An empty file or invalid
header raises `ValueError`; unreadable files raise an I/O error.

For well-shaped records, junctions are parsed as integers without coordinate
conversion. Zero, negative, and invalid junction counts are reported; these do
not establish a zero-based/one-based convention. Chromosome counts retain the
first 1,000 distinct names; subsequent unseen names contribute to an explicitly
reported overflow count. No chromosome-name normalization occurs.

For each available flag, `zero`, `nonzero`, `empty`, and `nonnumeric` are mutually
exclusive categories after whitespace trimming. Only finite numeric values are
numeric. The report additionally retains counts for the first 20 distinct raw
literal values; later unseen values contribute to `other_literal_rows`. This is
not a top-frequency estimate. Presence of a nonzero flag does not authorize
filtering. Histogram limits bound retained distinct values, not input cell size.

CLI: `python3 scripts/audit_tlx.py file.tlx [file.tlx.gz ...]` prints one JSON
object to stdout. All reports must succeed before JSON is printed. Invalid
arguments, input headers, corrupt compressed inputs, or decoding errors produce
a nonzero exit and a diagnostic on stderr. Inputs remain byte-for-byte unchanged.

Test matrix established before implementation: plain/gzip x LF/CRLF/CR;
header-only and missing/duplicate/empty headers; blank records, empty cells,
wrong-width rows; positive/zero/negative/invalid junctions; numeric/nonfinite/text
flags and missing optional flags; literal/chromosome overflow; one/multiple CLI
inputs and CLI failure; input-byte preservation. Fixtures are small artificial
parser examples, not biological validation or filtering provenance.
