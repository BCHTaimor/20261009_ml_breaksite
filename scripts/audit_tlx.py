#!/usr/bin/env python3
"""Stream TLX files read-only and print a compact JSON flag/coordinate audit.

Usage: python3 scripts/audit_tlx.py data/raw/breaksites/*.tlx
Requires only Python's standard library. Flag values are reported literally;
nonzero values are not interpreted as exclusion rules. No coordinates are
converted and no records are filtered or changed. Empty TSV cells are preserved.
"""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

FLAGS = ('unaligned', 'baitonly', 'uncut', 'misprimed', 'freqcut',
         'largegap', 'mapqual', 'breaksite', 'sequential', 'repeatseq', 'duplicate')


def audit(path):
    """Audit a plain or gzip TSV path read-only; return a JSON-compatible report.

    Requires a unique, nonempty header with Rname and Junction. Wrong-width
    records are reported, not interpreted. Integer coordinates are unchanged;
    flag categories carry no exclusion semantics. Retain at most 20 literal
    values per flag and 1,000 chromosome names, with explicit overflow counts.
    Raises ValueError for invalid headers; I/O/decoding errors propagate.
    See docs/tlx_audit_spec.md for denominator and CLI contracts.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    result = {'file': str(path), 'resolved_source': str(path.resolve()),
              'rows': 0, 'malformed_rows': 0, 'malformed_line_examples': [],
              'valid_integer_junctions': 0, 'invalid_junctions': 0,
              'zero_junctions': 0, 'negative_junctions': 0,
              'other_chromosome_rows': 0}
    chromosomes, flags = Counter(), {}
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader, [])
        header = [v.strip().lstrip('\ufeff') for v in header]
        if not {'Rname', 'Junction'}.issubset(header):
            raise ValueError(f'{path}: expected tab-delimited Rname and Junction columns; do not use wrapped terminal text')
        if len(set(header)) != len(header):
            raise ValueError(f'{path}: duplicate column names')
        if any(not name for name in header):
            raise ValueError(f'{path}: empty column name')
        result['columns'] = header
        result['missing_flag_columns'] = [key for key in FLAGS if key not in header]
        flags = {key: {'zero': 0, 'nonzero': 0, 'empty': 0, 'nonnumeric': 0,
                       'literal_values': {}, 'other_literal_rows': 0}
                 for key in FLAGS if key in header}
        for row in reader:
            if not row:
                continue
            result['rows'] += 1
            if len(row) != len(header):
                result['malformed_rows'] += 1
                if len(result['malformed_line_examples']) < 5:
                    result['malformed_line_examples'].append(reader.line_num)
                continue
            record = dict(zip(header, row))
            chromosome = record['Rname']
            if chromosome in chromosomes or len(chromosomes) < 1000:
                chromosomes[chromosome] += 1
            else:
                result['other_chromosome_rows'] += 1
            try:
                position = int(record['Junction'])
                result['valid_integer_junctions'] += 1
                result['zero_junctions'] += position == 0
                result['negative_junctions'] += position < 0
            except ValueError:
                result['invalid_junctions'] += 1
            for key, counts in flags.items():
                literal = record[key]
                literals = counts['literal_values']
                if literal in literals or len(literals) < 20:
                    literals[literal] = literals.get(literal, 0) + 1
                else:
                    counts['other_literal_rows'] += 1
                value = record[key].strip()
                if not value:
                    counts['empty'] += 1
                    continue
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError('nonfinite')
                    counts['zero' if number == 0 else 'nonzero'] += 1
                except ValueError:
                    counts['nonnumeric'] += 1
    result['chromosome_rows'] = dict(chromosomes)
    result['flags'] = flags
    return result


def main():
    """Print all requested reports as JSON; fail without partial JSON on error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.files:
        try:
            reports.append(audit(path))
        except (OSError, ValueError, csv.Error, EOFError) as exc:
            parser.error(str(exc))
    print(json.dumps({'files': reports, 'interpretation':
                     'Counts describe records only. Filtering rules, coordinate convention, '
                     'and normalization provenance still require the generating pipeline.'}, indent=2))


if __name__ == '__main__':
    main()
