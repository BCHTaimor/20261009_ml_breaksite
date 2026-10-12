"""Contract targets: read_tlx, groseq_depths, write_dataset, main.

Partitions: zero/one-based first/boundary/last/tail; gzip/tab-empty fields;
all rows/selected zero flags/malformed rows/off-reference/multiple sources;
BAM CIGAR D/N, primary/secondary/supplementary/duplicate/QC-fail, paired overlap;
BigWig zero/negative/full/partial/missing/out-of-range; multiple sizes and incomplete tails;
CLI required semantics/reference mismatch/no overwrite/manifest completion.
Invariants: immutable sources; exact hand-calculated counts; shared TSV schema.
These synthetic fixtures validate software, not biological performance.
"""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('builder', Path(__file__).parents[1] / 'scripts/build_shared_datasets.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
try:
    import pysam
    import pyBigWig
except ImportError:
    pysam = pyBigWig = None


class ExclusionContractTests(unittest.TestCase):
    def test_bed_merge_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mask.bed'
            path.write_text('# mask\nchr1\t8\t10\tlabel\nchr1\t4\t8\nchr1\t5\t9\n\n')
            self.assertEqual(builder.read_exclusions(path, {'chr1':20}), {'chr1':[(4,10)]})
            for row in ['chr2 1 2', 'chr1 -1 2', 'chr1 2 2', 'chr1 3 2',
                        'chr1 0 21', 'chr1 x 2', 'chr1 0']:
                path.write_text(row + '\n')
                with self.subTest(row=row), self.assertRaisesRegex(ValueError, ':1:'):
                    builder.read_exclusions(path, {'chr1':20})

    def test_overlap_omits_bins_and_preserves_true_zeros_at_both_sizes(self):
        class FullSignal:
            def stats(self, chrom, start, end, type, exact):
                return [1.0 if type == 'coverage' else 0.0]
        with tempfile.TemporaryDirectory() as directory:
            for size, expected_starts in [(5,[0,15]), (10,[])]:
                path = Path(directory) / f'{size}.tsv'
                report = builder.write_dataset(path, {'chr1':20}, size, FullSignal(),
                    {'chr1':[0]*(20//size)}, {'chr1':[0,99,88,0] if size == 5 else [99,88]},
                    'fixture', 'exclude-any-missing', {'chr1':[(5,11)]})
                with path.open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual([int(r['start']) for r in rows], expected_starts)
                self.assertTrue(all(int(r['break_count']) == 0 for r in rows))
                self.assertEqual(report['excluded_mask'], 2)
                self.assertEqual(report['excluded_missing_okseq'], 0)


class TlxContractTests(unittest.TestCase):
    def test_tab_empty_boundaries_flags_sources_and_tails(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'a.tlx'
            first.write_text('Qname\tRname\tJunction\tflag\n\tchr1\t0\t0\na\tchr1\t4\t1\nb\tchr1\t5\t0\nc\tchr1\t10\t0\nd\tother\t0\t0\n')
            second = Path(directory) / 'b.tlx.gz'
            with gzip.open(second, 'wt') as handle:
                handle.write('Rname\tJunction\tflag\nchr1\t9\t0\n')
            counts, reports = builder.read_tlx([first, second], {'chr1':11}, [5,10], 'zero-based', 'zero-flags', ['flag'])
            self.assertEqual(list(counts[5]['chr1']), [1,2])
            self.assertEqual(list(counts[10]['chr1']), [3])
            self.assertEqual(reports[0]['retained'], 3)
            self.assertEqual(reports[0]['outside_reference'], 1)
            self.assertEqual(reports[0]['filtered'], 1)
            all_counts, _ = builder.read_tlx([first], {'chr1':11}, [5], 'zero-based', 'all-rows', [])
            self.assertEqual(list(all_counts[5]['chr1']), [2,1])

    def test_one_based_and_malformed_records_fail_with_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'a.tlx'
            path.write_text('Rname\tJunction\nchr1\t1\nchr1\t5\nchr1\t6\nchr1\t10\n')
            counts, _ = builder.read_tlx([path], {'chr1':10}, [5], 'one-based', 'all-rows', [])
            self.assertEqual(list(counts[5]['chr1']), [2,2])
            for row in ['chr1\t0', 'chr1\t11', 'chr1\tx', 'chr1', 'chr1\t1\textra']:
                path.write_text('Rname\tJunction\n' + row + '\n')
                with self.subTest(row=row), self.assertRaisesRegex(ValueError, ':2:'):
                    builder.read_tlx([path], {'chr1':10}, [5], 'one-based', 'all-rows', [])
            path.write_text('Rname\tJunction\tflag\nchr1\t1\t\n')
            with self.assertRaisesRegex(ValueError, ':2:'):
                builder.read_tlx([path], {'chr1':10}, [5], 'one-based', 'zero-flags', ['flag'])

    def test_chromosome_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sizes'
            for text in ['', 'chr1 0\n', 'chr1 5\nchr1 5\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    builder.read_chrom_sizes(path)
            path.write_text('chr1 11\n')
            self.assertEqual(builder.read_chrom_sizes(path), {'chr1':11})


@unittest.skipUnless(pysam and pyBigWig, 'native readers not installed')
class NativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bam = self.root / 'reads.bam'
        with pysam.AlignmentFile(str(self.bam), 'wb', header={'HD':{'VN':'1.6'}, 'SQ':[{'SN':'chr1','LN':21}]}) as handle:
            # Aligned bp: first read [0,3),[5,7),[8,10) => 3,4,0,0.
            # Two overlapping paired alignments [4,8) => +2,+6,0,0.
            for name, start, cigar, flag in [('gaps',0,[(0,3),(3,2),(0,2),(2,1),(0,2)],0),
                ('pair',4,[(0,4)],65),('pair',4,[(0,4)],129),
                ('secondary',0,[(0,5)],256),('supplementary',0,[(0,5)],2048),
                ('duplicate',0,[(0,5)],1024),('qc',0,[(0,5)],512)]:
                read = pysam.AlignedSegment()
                read.query_name, read.reference_id, read.reference_start = name, 0, start
                read.flag, read.mapping_quality, read.cigartuples = flag, 60, cigar
                read.query_sequence = 'A' * sum(n for op,n in cigar if op in (0,1,4,7,8))
                handle.write(read)
        self.bw = self.root / 'signal.bw'
        with pyBigWig.open(str(self.bw), 'w') as handle:
            handle.addHeader([('chr1',21)])
            handle.addEntries(['chr1']*3, [0,5,10], ends=[5,10,12], values=[0.0,-0.5,0.8])
        self.tlx = self.root / 'junctions.tlx'
        self.tlx.write_text('Rname\tJunction\nchr1\t0\nchr1\t5\nchr1\t20\n')
        self.sizes = self.root / 'sizes'
        self.sizes.write_text('chr1\t21\n')

    def test_bam_gaps_primary_filter_and_overlapping_mates_multiple_sizes(self):
        with pysam.AlignmentFile(str(self.bam), 'rb') as handle:
            depths, audit = builder.groseq_depths(handle, {'chr1':21}, [5,10])
        self.assertEqual(list(depths[5]['chr1']), [5,10,0,0])
        self.assertEqual(list(depths[10]['chr1']), [15,0])
        self.assertEqual(audit['filtered'], 4)
        self.assertEqual(audit['retained'], 3)

    def test_bigwig_missing_partial_and_real_zero(self):
        with pysam.AlignmentFile(str(self.bam), 'rb') as bam, pyBigWig.open(str(self.bw)) as bw:
            depths, _ = builder.groseq_depths(bam, {'chr1':21}, [5])
            counts, _ = builder.read_tlx([self.tlx], {'chr1':21}, [5], 'zero-based', 'all-rows', [])
            for policy, expected in [('exclude-any-missing',2), ('covered-mean',3)]:
                path = self.root / (policy + '.tsv')
                audit = builder.write_dataset(path, {'chr1':21}, 5, bw, depths[5], counts[5], 'fixture', policy)
                with path.open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertEqual(audit['written'], expected)
                self.assertEqual([float(r['okseq']) for r in rows[:2]], [0,-0.5])
                self.assertEqual([float(r['groseq']) for r in rows[:2]], [1,2])
                self.assertEqual([int(r['break_count']) for r in rows[:2]], [1,1])
                with self.assertRaises(FileExistsError):
                    builder.write_dataset(path, {'chr1':21}, 5, bw, depths[5], counts[5], 'fixture', policy)

    def arguments(self):
        return ['--chrom-sizes',str(self.sizes),'--okseq',str(self.bw),'--groseq-bam',str(self.bam),
                '--breaksite',str(self.tlx),'--output-root',str(self.root/'output'),
                '--bin-sizes','5','10','--genome-assembly','fixture',
                '--tlx-coordinate-system','zero-based','--tlx-filter-policy','all-rows',
                '--label-policy','pooled-retained-rows','--missing-okseq-policy','exclude-any-missing']

    def test_cli_complete_manifest_hashes_and_no_overwrite(self):
        builder.main(self.arguments())
        manifest = json.loads((self.root/'output/build_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual([x['written'] for x in manifest['outputs']], [2,1])
        self.assertTrue(all(len(x['sha256']) == 64 for x in manifest['outputs']))
        before = (self.root/'output/bins_5bp/dataset.tsv').read_bytes()
        with self.assertRaises(FileExistsError):
            builder.main(self.arguments())
        self.assertEqual((self.root/'output/bins_5bp/dataset.tsv').read_bytes(), before)

    def test_cli_reference_mismatch_and_required_semantics(self):
        self.sizes.write_text('chr1\t22\n')
        with self.assertRaisesRegex(ValueError, 'Reference mismatch'):
            builder.main(self.arguments())
        with self.assertRaises(SystemExit):
            builder.main(self.arguments()[:-2])

    def test_cli_mask_identity_and_counts(self):
        mask = self.root / 'mask.bed'
        mask.write_text('chr1\t5\t6\n')
        builder.main(self.arguments() + ['--exclude-bed', str(mask)])
        manifest = json.loads((self.root/'output/build_manifest.json').read_text())
        self.assertEqual(manifest['exclusion_mask']['path'], str(mask))
        self.assertEqual(len(manifest['exclusion_mask']['sha256']), 64)
        self.assertEqual([x['excluded_mask'] for x in manifest['outputs']], [1,1])
        self.assertEqual([x['written'] for x in manifest['outputs']], [1,0])

    def test_invalid_rfd_records_failed_manifest(self):
        invalid = self.root / 'invalid.bw'
        with pyBigWig.open(str(invalid), 'w') as handle:
            handle.addHeader([('chr1',21)])
            handle.addEntries(['chr1'], [0], ends=[21], values=[2.0])
        args = self.arguments()
        args[args.index('--okseq') + 1] = str(invalid)
        with self.assertRaisesRegex(ValueError, 'RFD outside'):
            builder.main(args)
        manifest = json.loads((self.root/'output/build_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
