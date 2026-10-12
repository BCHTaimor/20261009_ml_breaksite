# Contract target: audit_tlx.audit and audit_tlx CLI; docs/tlx_audit_spec.md.
# Partitions: plain/gzip x LF/CRLF/CR; header-only/missing/duplicate/empty;
# blank/wrong-width/empty-cell rows; integer zero/negative/positive/invalid;
# flags numeric/nonfinite/text/missing; bounded literal/chromosome overflow;
# CLI one/multiple inputs, invalid input, corrupt gzip, missing arguments.
# Invariants: source bytes unchanged; category totals equal well-shaped rows;
# no coordinate conversion/filtering; failed CLI emits no partial JSON.
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_tlx.py'
SPEC = importlib.util.spec_from_file_location('audit_tlx', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TlxAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, content, name='input.tlx', compressed=False):
        path = self.root / name
        payload = content.encode('utf-8')
        path.write_bytes(gzip.compress(payload) if compressed else payload)
        return path

    def test_encodings_and_newlines_preserve_empty_cells_and_original_bytes(self):
        for compressed in (False, True):
            for newline in ('\n', '\r\n', '\r'):
                with self.subTest(compressed=compressed, newline=repr(newline)):
                    path = self.fixture(newline.join([
                        '\ufeffRname\tJunction\tduplicate\tuncut',
                        'chr1\t0\t\t1', 'chr2\t-1\t0\t', 'chr1\t12\t2\t0',
                        'chr3\tbad\tNaN\tfalse', '',
                    ]), 'input.tlx.gz' if compressed else 'input.tlx', compressed)
                    original = path.read_bytes()
                    report = MODULE.audit(path)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(report['rows'], 4)
                    self.assertEqual(report['valid_integer_junctions'], 3)
                    self.assertEqual(report['zero_junctions'], 1)
                    self.assertEqual(report['negative_junctions'], 1)
                    self.assertEqual(report['invalid_junctions'], 1)
                    self.assertEqual(report['chromosome_rows'], {'chr1': 2, 'chr2': 1, 'chr3': 1})
                    for flag in ('duplicate', 'uncut'):
                        self.assertEqual([report['flags'][flag][k] for k in
                                          ('zero', 'nonzero', 'empty', 'nonnumeric')], [1, 1, 1, 1])
                    self.assertIn('misprimed', report['missing_flag_columns'])

    def test_header_only_and_optional_flags_absent(self):
        report = MODULE.audit(self.fixture('Rname\tJunction\n'))
        self.assertEqual(report['rows'], 0)
        self.assertEqual(report['flags'], {})

    def test_invalid_headers_fail_explicitly(self):
        for header in ('', 'Rname Junction', 'Rname\tOther',
                       'Rname\tJunction\tRname', 'Rname\tJunction\t'):
            with self.subTest(header=header), self.assertRaises(ValueError):
                MODULE.audit(self.fixture(header + '\n'))

    def test_malformed_denominator_and_bounded_line_examples(self):
        path = self.fixture('Rname\tJunction\tduplicate\n\n' +
                            'chr1\t2\n' * 6 + 'chr1\t3\t0\n')
        report = MODULE.audit(path)
        self.assertEqual(report['rows'], 7)
        self.assertEqual(report['malformed_rows'], 6)
        self.assertEqual(report['malformed_line_examples'], [3, 4, 5, 6, 7])
        self.assertEqual(report['valid_integer_junctions'], 1)
        self.assertEqual(report['flags']['duplicate']['zero'], 1)

    def test_literal_numeric_semantics_and_histogram_overflow(self):
        literals = [' 0 ', '-0', '1e2', '-3', 'inf', '-inf', 'NaN', 'text', '']
        literals += [str(i) for i in range(30)]
        report = MODULE.audit(self.fixture('Rname\tJunction\tduplicate\n' +
                     ''.join('chr1\t1\t' + value + '\n' for value in literals)))
        flag = report['flags']['duplicate']
        self.assertEqual(flag['zero'], 3)
        self.assertEqual(flag['nonzero'], 31)
        self.assertEqual(flag['nonnumeric'], 4)
        self.assertEqual(flag['empty'], 1)
        self.assertEqual(len(flag['literal_values']), 20)
        self.assertEqual(sum(flag['literal_values'].values()) + flag['other_literal_rows'], len(literals))
        self.assertEqual(flag['literal_values'][' 0 '], 1)

    def test_chromosome_overflow_is_explicit(self):
        report = MODULE.audit(self.fixture('Rname\tJunction\n' +
                     ''.join(f'contig{i}\t1\n' for i in range(1002)) + 'contig0\t1\n'))
        self.assertEqual(len(report['chromosome_rows']), 1000)
        self.assertEqual(report['other_chromosome_rows'], 2)
        self.assertEqual(report['chromosome_rows']['contig0'], 2)

    def test_cli_multiple_inputs_and_failure_no_partial_json(self):
        path = self.fixture('Rname\tJunction\nchr1\t0\n')
        second = self.fixture('Rname\tJunction\n', 'second.tlx.gz', True)
        run = subprocess.run([sys.executable, str(SCRIPT), str(path), str(second)],
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(json.loads(run.stdout)['files']), 2)
        broken = self.fixture('invalid', 'broken.tlx.gz')
        for args in ([], [str(path), str(broken)], [str(self.root / 'missing.tlx')]):
            with self.subTest(args=args):
                run = subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, '')
                self.assertIn('error:', run.stderr)


if __name__ == '__main__':
    unittest.main()
