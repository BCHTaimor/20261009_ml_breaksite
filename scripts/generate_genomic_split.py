#!/usr/bin/env python3
r"""Generate reproducible chromosome-wise random train/test regions.

Example (supply the verified assembly's chromosome sizes and gene convention):
    python3 scripts/generate_genomic_split.py \
        --chrom-sizes data/reference/chrom.sizes --genome-build VERIFIED_ASSEMBLY \
        --gene-coordinates one-based-closed --seed 20261009 \
        --output-dir splits/seed_20261009

Omit --seed to generate a random seed; omit --output-dir to use splits/seed_<seed>.
Writes manifest.json plus train/test BEDs and gene_assignments.tsv, never data copies.

Chromosome sizes: headerless chromosome<TAB>length. Include only chromosomes
intended for modeling; every listed chromosome must support both partitions.
Genes: headerless chromosome<TAB>start<TAB>end<TAB>gene. Output is always BED
(zero-based, half-open). No assembly or coordinate convention is inferred.

5 Mb blocks crossed by an RDC gene are joined. For each chromosome, randomized
orders of these indivisible groups produce candidate train subsets; choose the
prefix closest to the requested genomic span, breaking ties randomly. This is
constrained random sampling, not uniform sampling over all possible subsets.
Reject complete draws that do not satisfy RDC counts. Reuse the resulting regions
for every prediction-bin size. Outputs cover all supplied chromosome spans;
they DO NOT apply mappability masks, boundary leakage buffers, or binning. Apply
appropriate boundary exclusions before fitting models with smoothed features.
Requires Python 3.9+ and the standard library only. Refuses existing output dirs.
"""

import argparse
import hashlib
import json
import platform
import random
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_inputs(chrom_path, gene_path, convention):
    """Read and validate sizes and unique genes; return BED-normalized records."""
    sizes = {}
    for number, line in enumerate(chrom_path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{chrom_path}:{number}: expected chromosome and length')
        chrom, length = fields[0], int(fields[1])
        if chrom in sizes or length <= 0:
            raise ValueError(f'{chrom_path}:{number}: duplicate chromosome or invalid length')
        sizes[chrom] = length
    if not sizes:
        raise ValueError('Chromosome sizes are empty')
    genes, names = [], set()
    for number, line in enumerate(gene_path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f'{gene_path}:{number}: expected chromosome/start/end/gene')
        chrom, start, end, name = fields
        start, end = int(start), int(end)
        if convention == 'one-based-closed':
            start -= 1
        if chrom not in sizes or not 0 <= start < end <= sizes[chrom]:
            raise ValueError(f'{gene_path}:{number}: invalid coordinates or missing chromosome')
        if name in names:
            raise ValueError(f'Duplicate gene: {name}')
        names.add(name)
        genes.append((chrom, start, end, name))
    if not genes:
        raise ValueError('Gene file is empty')
    return sizes, genes


def make_groups(sizes, genes, block_size):
    """Return contiguous groups; remove any nominal boundary inside an RDC gene."""
    groups = {}
    for chrom, length in sizes.items():
        intervals = [(start, end) for c, start, end, _ in genes if c == chrom]
        boundaries = [0] + [position for position in range(block_size, length, block_size)
                            if not any(start < position < end for start, end in intervals)] + [length]
        groups[chrom] = list(zip(boundaries, boundaries[1:]))
        if len(groups[chrom]) < 2:
            raise ValueError(f'{chrom}: fewer than two independent groups; cannot put it in both sets')
    return groups


def sample_chromosome(groups, fraction, trials, rng):
    """Choose a random-subset candidate closest to target span; both sets nonempty."""
    target = sum(end - start for start, end in groups) * fraction
    best_error, winner, ties = float('inf'), None, 0
    for _ in range(trials):
        order = list(range(len(groups)))
        rng.shuffle(order)
        span = 0
        for k, index in enumerate(order[:-1], 1):
            span += groups[index][1] - groups[index][0]
            error = abs(span - target)
            if error < best_error:
                best_error, winner, ties = error, set(order[:k]), 1
            elif error == best_error:
                ties += 1
                if rng.randrange(ties) == 0:
                    winner = set(order[:k])
    return [(start, end, 'train' if index in winner else 'test')
            for index, (start, end) in enumerate(groups)]


def assign_genes(regions, genes):
    """Return gene assignments; fail if a gene is not contained in exactly one group."""
    assignments = []
    for chrom, start, end, name in genes:
        matches = [label for left, right, label in regions[chrom] if left <= start and end <= right]
        if len(matches) != 1:
            raise ValueError(f'Gene is split or missing: {name}')
        assignments.append((chrom, start, end, name, matches[0]))
    return assignments


def validate_regions(regions, sizes):
    """Enforce exact disjoint coverage and both partitions on each chromosome."""
    for chrom, length in sizes.items():
        cursor, labels = 0, set()
        for start, end, label in regions[chrom]:
            if start != cursor or end <= start or label not in {'train', 'test'}:
                raise ValueError(f'Invalid region partition on {chrom}')
            cursor = end
            labels.add(label)
        if cursor != length or labels != {'train', 'test'}:
            raise ValueError(f'Incomplete partition on {chrom}')


def main(argv=None):
    """Validate CLI inputs, generate a constrained split, and write a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chrom-sizes', type=Path, required=True)
    parser.add_argument('--genes', type=Path, default=Path('data/reference/u2os_rdc_genes_matched34.txt'))
    parser.add_argument('--gene-coordinates', choices=['bed', 'one-based-closed'], required=True)
    parser.add_argument('--genome-build', required=True, help='Verified assembly name recorded as metadata')
    parser.add_argument('--output-dir', type=Path, help='Default: splits/seed_<seed>; never overwritten')
    parser.add_argument('--coordinate-note', default='User-specified convention; not independently verified')
    parser.add_argument('--block-size', type=int, default=5_000_000)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--min-train-rdc', type=int, default=17)
    parser.add_argument('--min-test-rdc', type=int, default=1)
    parser.add_argument('--seed', type=int, help='Omit to generate and record a random 63-bit seed')
    parser.add_argument('--chromosome-trials', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = secrets.randbits(63)
    if args.output_dir is None:
        args.output_dir = Path('splits') / f'seed_{args.seed}'
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError(f'Output directory already exists: {args.output_dir}')
        if args.block_size <= 0 or not 0 < args.train_fraction < 1:
            raise ValueError('Block size must be positive and train fraction strictly between zero and one')
        if min(args.chromosome_trials, args.max_attempts) < 1 or min(args.min_train_rdc, args.min_test_rdc) < 0:
            raise ValueError('Trials/attempts must be positive and RDC minima nonnegative')
        sizes, genes = read_inputs(args.chrom_sizes, args.genes, args.gene_coordinates)
        if args.min_train_rdc + args.min_test_rdc > len(genes):
            raise ValueError('RDC minima exceed the available gene count')
        groups = make_groups(sizes, genes, args.block_size)
        rng = random.Random(args.seed)
        for attempt in range(1, args.max_attempts + 1):
            regions = {chrom: sample_chromosome(items, args.train_fraction, args.chromosome_trials, rng)
                       for chrom, items in groups.items()}
            assignments = assign_genes(regions, genes)
            train_count = sum(row[4] == 'train' for row in assignments)
            if train_count >= args.min_train_rdc and len(genes) - train_count >= args.min_test_rdc:
                break
        else:
            raise ValueError('No split satisfied RDC constraints within --max-attempts; no outputs written')
        validate_regions(regions, sizes)
        per_chromosome = {}
        for chrom, items in regions.items():
            train_bp = sum(end - start for start, end, label in items if label == 'train')
            per_chromosome[chrom] = {'length_bp': sizes[chrom], 'train_bp': train_bp,
                                     'test_bp': sizes[chrom] - train_bp, 'train_fraction': train_bp / sizes[chrom],
                                     'independent_groups': len(items)}
        summary = {
            'schema_version': 1,
            'storage_design': 'shared-tsv-region-only-splits',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'coordinate_convention_status': args.coordinate_note,
            'bin_selection': {
                'rule': 'Full bin containment in the union of adjacent same-partition regions',
                'cross_partition_bins': 'exclude',
                'boundary_buffer_bp': None,
                'status': 'Boundary buffer must be specified in model run config before fitting',
            },
            'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            'algorithm': 'randomized prefix candidates per chromosome; closest span; random ties; RDC rejection',
            'accepted_attempt': attempt, 'python_version': platform.python_version(),
            'command': [sys.executable, *sys.argv] if argv is None else list(argv),
            'inputs': {label: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for label, path in [('genes', args.genes), ('chrom_sizes', args.chrom_sizes)]},
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gene_counts': {'total': len(genes), 'train': train_count, 'test': len(genes) - train_count},
            'chromosomes': per_chromosome, 'output_coordinates': 'zero-based-half-open',
            'boundary_buffers_applied': False, 'mappability_masks_applied': False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for label in ['train', 'test']:
            with (args.output_dir / f'{label}_regions.bed').open('x') as handle:
                for chrom, items in regions.items():
                    for start, end, partition in items:
                        if partition == label:
                            handle.write(f'{chrom}\t{start}\t{end}\n')
        with (args.output_dir / 'gene_assignments.tsv').open('x') as handle:
            handle.write('chrom\tstart\tend\tgene\tpartition\n')
            for row in assignments:
                handle.write('\t'.join(map(str, row)) + '\n')
        with (args.output_dir / 'manifest.json').open('x') as handle:
            json.dump(summary, handle, indent=2)
            handle.write('\n')
        print(f'Wrote {args.output_dir}: {train_count} training RDC genes, {len(genes) - train_count} testing RDC genes')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
