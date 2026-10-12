"""Load shared TSVs using genomic partitions, without writing per-seed tables."""
import csv
import math
import random
from pathlib import Path

import numpy as np


def load_regions(split_dir, validation_fraction, seed):
    """Return sorted, merged (chrom,start,end,partition) tuples.

    Randomly hold out whole training BED rows for validation, before merging.
    The generated split's rows are indivisible RDC-preserving groups. Testing
    rows are never used to select validation. Requires disjoint full chromosome
    coverage and at least two training groups. No files are changed.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    rows = []
    for label in ('train', 'test'):
        with (Path(split_dir) / f'{label}_regions.bed').open() as handle:
            for line in handle:
                if not line.strip() or line.startswith('#'):
                    continue
                chrom, start, end = line.split()[:3]
                start, end = int(start), int(end)
                if not 0 <= start < end:
                    raise ValueError('Invalid BED region')
                rows.append((chrom, start, end, label))
    rows.sort()
    if not rows or {r[3] for r in rows} != {'train', 'test'}:
        raise ValueError('Both train and test regions are required')
    ends = {}
    for chrom, start, end, _ in rows:
        if start != ends.get(chrom, 0):
            raise ValueError('Regions must cover each chromosome without gaps or overlap')
        ends[chrom] = end
    candidates = [i for i, r in enumerate(rows) if r[3] == 'train']
    if len(candidates) < 2:
        raise ValueError('At least two training groups needed for validation')
    random.Random(seed).shuffle(candidates)
    target = sum(rows[i][2] - rows[i][1] for i in candidates) * validation_fraction
    cumulative = np.cumsum([rows[i][2] - rows[i][1] for i in candidates[:-1]])
    n = int(np.argmin(np.abs(cumulative - target))) + 1
    for i in candidates[:n]:
        rows[i] = (*rows[i][:3], 'validation')
    merged = []
    for row in rows:
        if merged and merged[-1][0] == row[0] and merged[-1][2] == row[1] and merged[-1][3] == row[3]:
            merged[-1] = (row[0], merged[-1][1], row[2], row[3])
        else:
            merged.append(row)
    return merged


def classify_bin(chrom, start, end, regions, buffer_bp):
    """Return membership after full containment and symmetric boundary exclusions.

    Regions must be sorted and adjacent same-partition intervals merged. Trim
    buffer_bp from each side of every partition transition, not chromosome ends.
    Bins outside the split or crossing a boundary return 'excluded'.
    """
    if buffer_bp < 0 or not 0 <= start < end:
        raise ValueError('Invalid buffer or interval')
    for i, (c, left, right, label) in enumerate(regions):
        if c != chrom:
            continue
        lo = left + (buffer_bp if i > 0 and regions[i-1][0] == c else 0)
        hi = right - (buffer_bp if i+1 < len(regions) and regions[i+1][0] == c else 0)
        if lo <= start and end <= hi:
            return label
    return 'excluded'


def load_datasets(paths, regions, buffer_bp):
    """Read raw-count TSVs into arrays; no disk copies or learned transforms.

    Required columns: bin_id, chrom, start, end, bin_size_bp, okseq, groseq,
    break_count. Inputs must be finite; RFD is [-1,1], GRO-seq nonnegative,
    targets nonnegative integers. Bins must be full-sized, aligned to zero,
    uniquely identified and nonoverlapping within each chromosome/resolution.
    Mixed resolutions share region assignments; excluded rows remain marked.
    """
    required = {'bin_id','chrom','start','end','bin_size_bp','okseq','groseq','break_count'}
    rows, x, y, labels = [], [], [], []
    seen_ids, seen_coords = set(), set()
    by_chrom = {}
    for r in regions:
        by_chrom.setdefault(r[0], []).append(r)
    for path in paths:
        with Path(path).open() as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path}: missing required TSV columns')
            for line, row in enumerate(reader, 2):
                try:
                    chrom, bid = row['chrom'], row['bin_id']
                    start, end, size = (int(row[k]) for k in ('start','end','bin_size_bp'))
                    ok, gro, count = (float(row[k]) for k in ('okseq','groseq','break_count'))
                    if not bid or not chrom or not all(math.isfinite(v) for v in (ok,gro,count)):
                        raise ValueError('Missing identifiers or nonfinite values')
                    if not (-1 <= ok <= 1 and gro >= 0 and count >= 0 and count.is_integer()):
                        raise ValueError('Expected RFD [-1,1], nonnegative GRO-seq, and raw integer counts')
                    if size <= 0 or start < 0 or end-start != size or start % size:
                        raise ValueError('Expected full-sized bins on a chromosome-zero grid')
                    key = (chrom,start,end)
                    if bid in seen_ids or key in seen_coords:
                        raise ValueError('Duplicate bin identifier or coordinates')
                    seen_ids.add(bid); seen_coords.add(key)
                    part = classify_bin(chrom,start,end,by_chrom.get(chrom,[]),buffer_bp)
                    rows.append(row); x.append((ok,gro,size)); y.append(count); labels.append(part)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path}:{line}: {exc}') from exc
    if not rows:
        raise ValueError('No dataset rows')
    return {'x':np.asarray(x,dtype=np.float64),'y':np.asarray(y,dtype=np.float64),
            'partition':np.asarray(labels),'rows':rows}
