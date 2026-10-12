#!/usr/bin/env python3
"""Explicit, immutable shared dataset builder; contract: docs/dataset_builder_spec.md."""
import argparse
from array import array
from bisect import bisect_left
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def read_chrom_sizes(path):
    """Read unique positive chromosome lengths; reject malformed or empty input."""
    sizes = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] in sizes or int(fields[1]) <= 0:
            raise ValueError(f'{path}:{number}: expected unique chromosome and positive length')
        sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError('Empty chromosome sizes')
    return sizes


def _arrays(sizes, bin_sizes):
    return {b: {c: array('Q', [0]) * (n // b) for c, n in sizes.items()} for b in bin_sizes}


def read_exclusions(path, sizes):
    """Validate a BED against reference sizes and return merged intervals.

    Input is zero-based half-open; extra columns and blank/comment lines are
    allowed. Invalid rows raise ValueError with source/line context. Sources
    are read only; no exclusions are inferred from counts or flag names.
    """
    intervals = {}
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            try:
                chrom, left, right = line.split()[:3]
                left, right = int(left), int(right)
                if chrom not in sizes or not 0 <= left < right <= sizes[chrom]:
                    raise ValueError('unknown chromosome or invalid BED bounds')
            except ValueError as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
            intervals.setdefault(chrom, []).append((left, right))
    merged = {}
    for chrom, rows in intervals.items():
        result = []
        for left, right in sorted(rows):
            if result and left <= result[-1][1]:
                result[-1] = (result[-1][0], max(right, result[-1][1]))
            else:
                result.append((left, right))
        merged[chrom] = result
    return merged


def read_tlx(paths, sizes, bin_sizes, coordinate_system, filter_policy, zero_flags):
    """Return pooled per-bin counts and source audits; preserve tab-empty cells.

    Malformed rows fail with line context. Only explicitly selected flags filter
    rows. Each retained row contributes once, without weighting/deduplication.
    Off-reference chromosomes and terminal partial bins are audited and omitted.
    """
    counts, reports = _arrays(sizes, bin_sizes), []
    for path in paths:
        report = dict(path=str(path), rows=0, filtered=0, outside_reference=0,
                      retained=0, binned={str(b): 0 for b in bin_sizes})
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            header = next(reader, [])
            required = ['Rname', 'Junction'] + list(zero_flags)
            if len(set(header)) != len(header) or not set(required).issubset(header):
                raise ValueError(f'{path}: missing required or duplicate TLX header columns')
            indices = {name: header.index(name) for name in required}
            for values in reader:
                report['rows'] += 1
                try:
                    if len(values) != len(header):
                        raise ValueError('row width differs from header')
                    chrom, pos = values[indices['Rname']], int(values[indices['Junction']])
                    if not chrom or pos < (1 if coordinate_system == 'one-based' else 0):
                        raise ValueError('empty chromosome or invalid Junction')
                    pos -= coordinate_system == 'one-based'
                    flags = [float(values[indices[f]]) for f in zero_flags]
                    if not all(math.isfinite(f) for f in flags):
                        raise ValueError('nonfinite selected flag')
                    if chrom in sizes and pos >= sizes[chrom]:
                        raise ValueError('Junction outside selected chromosome bounds')
                except ValueError as exc:
                    raise ValueError(f'{path}:{reader.line_num}: {exc}') from exc
                if filter_policy == 'zero-flags' and any(flags):
                    report['filtered'] += 1
                elif chrom not in sizes:
                    report['outside_reference'] += 1
                else:
                    report['retained'] += 1
                    for b in bin_sizes:
                        if pos // b < len(counts[b][chrom]):
                            counts[b][chrom][pos // b] += 1
                            report['binned'][str(b)] += 1
        reports.append(report)
    return counts, reports


def groseq_depths(bam, sizes, bin_sizes):
    """Stream mapped primary, nonduplicate, QC-passing BAM reads once.

    Return aligned-base sums and audit. CIGAR D/N gaps contribute no bases;
    overlapping mates count independently, both strands, no quality threshold.
    """
    totals = _arrays(sizes, bin_sizes)
    report = dict(rows=0, filtered=0, outside_reference=0, retained=0)
    for read in bam.fetch(until_eof=True):
        report['rows'] += 1
        if read.flag & (4 | 256 | 512 | 1024 | 2048):
            report['filtered'] += 1
            continue
        chrom = read.reference_name
        if chrom not in sizes:
            report['outside_reference'] += 1
            continue
        report['retained'] += 1
        for start, end in read.get_blocks():
            if start < 0 or end > sizes[chrom]:
                raise ValueError(f'BAM alignment out of bounds: {read.query_name}')
            if end <= start:
                continue
            for b in bin_sizes:
                for index in range(start // b, min((end - 1) // b + 1, len(totals[b][chrom]))):
                    totals[b][chrom][index] += min(end, (index + 1) * b) - max(start, index * b)
    return totals, report


def write_dataset(path, sizes, bin_size, bw, depths, breaks, assembly, missing_policy, exclusions=None):
    """Exclusively write TSV with exact BigWig means and aligned bp/bin length.

    Entirely missing signal is excluded. Partial signal is retained only with
    covered-mean policy. Zero/negative RFD remains valid. Return bin audit.
    """
    report = dict(written=0, excluded_missing_okseq=0, excluded_mask=0)
    with path.open('x', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['bin_id','chrom','start','end','bin_size_bp','okseq','groseq','break_count'])
        for chrom, length in sizes.items():
            masked = (exclusions or {}).get(chrom, [])
            mask_starts = [left for left, _ in masked]
            for index, start in enumerate(range(0, length - bin_size + 1, bin_size)):
                end = start + bin_size
                last = bisect_left(mask_starts, end) - 1
                if last >= 0 and masked[last][1] > start:
                    report['excluded_mask'] += 1
                    continue
                coverage = bw.stats(chrom, start, end, type='coverage', exact=True)[0]
                value = bw.stats(chrom, start, end, type='mean', exact=True)[0]
                if value is not None and math.isfinite(value) and not -1 <= value <= 1:
                    raise ValueError(f'RFD outside [-1, 1]: {chrom}:{start}-{end}: {value}')
                if (coverage is None or value is None or not math.isfinite(value) or
                    not math.isfinite(coverage) or coverage <= 0 or
                    (missing_policy == 'exclude-any-missing' and coverage < 1)):
                    report['excluded_missing_okseq'] += 1
                    continue
                writer.writerow([f'{assembly}:{chrom}:{start}-{end}', chrom, start, end,
                                 bin_size, value, depths[chrom][index] / bin_size, breaks[chrom][index]])
                report['written'] += 1
    return report


def _identity(path, hashed=False):
    stat = path.stat()
    result = dict(path=str(path), resolved=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if hashed:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result['sha256'] = digest.hexdigest()
    return result


def main(argv=None):
    """Validate policies, create tables and completion manifest; refuse overwrite.

    Failed runs retain partial output and a failed manifest for inspection.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('chrom-sizes', 'okseq', 'groseq-bam'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--breaksite', type=Path, action='append', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('datasets'))
    parser.add_argument('--bin-sizes', type=int, nargs='+', default=[5000, 10000])
    parser.add_argument('--genome-assembly', required=True)
    parser.add_argument('--tlx-coordinate-system', choices=['zero-based','one-based'], required=True)
    parser.add_argument('--tlx-filter-policy', choices=['all-rows','zero-flags'], required=True)
    parser.add_argument('--zero-flag', action='append', default=[])
    parser.add_argument('--label-policy', choices=['pooled-retained-rows'], required=True)
    parser.add_argument('--missing-okseq-policy', choices=['exclude-any-missing','covered-mean'], required=True)
    parser.add_argument('--exclude-bed', type=Path,
                        help='Explicit zero-based half-open mask; omit whole overlapping bins')
    args = parser.parse_args(argv)
    if min(args.bin_sizes) <= 0 or len(set(args.bin_sizes)) != len(args.bin_sizes):
        parser.error('bin sizes must be positive and unique')
    if bool(args.zero_flag) != (args.tlx_filter_policy == 'zero-flags'):
        parser.error('--zero-flag must be supplied only with zero-flags policy')
    if len({p.resolve() for p in args.breaksite}) != len(args.breaksite):
        parser.error('duplicate TLX source path')
    outputs = {b: args.output_root / (f'bins_{b // 1000}kb' if b % 1000 == 0 else f'bins_{b}bp') / 'dataset.tsv'
               for b in args.bin_sizes}
    manifest_path = args.output_root / 'build_manifest.json'
    for path in [manifest_path, *outputs.values()]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing existing output: {path}; choose a new --output-root')
    sizes = read_chrom_sizes(args.chrom_sizes)
    exclusions = read_exclusions(args.exclude_bed, sizes) if args.exclude_bed else {}
    import pysam
    import pyBigWig
    with pysam.AlignmentFile(str(args.groseq_bam), 'rb') as bam, pyBigWig.open(str(args.okseq)) as bw:
        bam_sizes, bw_sizes = dict(zip(bam.references, bam.lengths)), bw.chroms()
        for chrom, length in sizes.items():
            if bam_sizes.get(chrom) != length or bw_sizes.get(chrom) != length:
                raise ValueError(f'Reference mismatch/missing chromosome: {chrom}')
        manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
            command=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            settings={k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k == 'breaksite' else v for k,v in vars(args).items()},
            software=dict(python=sys.version, pysam=pysam.__version__, pyBigWig=pyBigWig.__version__),
            sources=[_identity(p, p not in (args.groseq_bam, args.okseq)) for p in [args.chrom_sizes,args.okseq,args.groseq_bam,*args.breaksite]],
            script=_identity(Path(__file__), True))
        manifest['exclusion_mask'] = _identity(args.exclude_bed, True) if args.exclude_bed else None
        manifest['exclusion_rule'] = 'Omit whole bins overlapping explicit BED; retain valid zero counts'
        args.output_root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('x') as handle:
            json.dump(manifest, handle, indent=2)
        try:
            breaks, manifest['tlx_audit'] = read_tlx(args.breaksite, sizes, args.bin_sizes,
                args.tlx_coordinate_system, args.tlx_filter_policy, args.zero_flag)
            print('TLX counting complete; streaming BAM once.', flush=True)
            depths, manifest['bam_audit'] = groseq_depths(bam, sizes, args.bin_sizes)
            manifest['outputs'] = []
            for b, path in outputs.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                summary = write_dataset(path, sizes, b, bw, depths[b], breaks[b], args.genome_assembly, args.missing_okseq_policy, exclusions)
                manifest['outputs'].append(dict(**_identity(path, True), **summary))
                print(f'Wrote {path}: {summary}', flush=True)
            manifest['status'] = 'complete'
        except BaseException as exc:
            manifest['status'], manifest['error'] = 'failed', f'{type(exc).__name__}: {exc}'
            raise
        finally:
            manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
            with manifest_path.open('w') as handle:
                json.dump(manifest, handle, indent=2)


if __name__ == '__main__':
    main()
