"""Contract tests for the separate contextual transformer module."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_context_transformer as context
from ml_data import load_datasets, load_regions


class ContextTransformerContracts(unittest.TestCase):
    def test_model_shape_and_probability_correction(self):
        model = context.ContextTransformer(width=8, layers=1, heads=2,
                                           feedforward=16, dropout=0)
        logits = model(torch.zeros(3, 5, 6))
        self.assertEqual(tuple(logits.shape), (3,))
        corrected = context.corrected_probability(torch.tensor([np.log(9.)]), 9.)
        self.assertAlmostEqual(corrected.item(), .5)

    def test_windows_keep_zero_centers_masks_and_partitions(self):
        rows = []
        for index in range(7):
            rows.append({'bin_id': f'b{index}', 'chrom': 'chr1',
                         'start': str(index * 10), 'end': str((index + 1) * 10),
                         'bin_size_bp': '10', 'okseq': str(index / 10),
                         'groseq': str(index), 'break_count': str(index == 3)})
        data = {'rows': rows, 'x': np.array([[i / 10, i, 10] for i in range(7)]),
                'y': np.array([0, 0, 0, 1, 0, 0, 0], dtype=float),
                'partition': np.array(['train'] * 5 + ['test'] * 2)}
        grid = {(row['chrom'], int(row['start'])):
                (float(row['okseq']), float(row['groseq'])) for row in rows}
        grid[('chr1', 20)] = (float('nan'), 2.)
        regions = [('chr1', 0, 50, 'train'), ('chr1', 50, 70, 'test')]
        mean, scale = context.fit_scaler(data['x'][:5])
        windows = context.build_windows(data, grid, regions, 3, 0, mean, scale)
        self.assertIn(0., windows['y'])
        center_three = int(np.where(windows['row_index'] == 3)[0][0])
        self.assertEqual(windows['x'][center_three, 0, 2], 0.)
        self.assertEqual(windows['x'][center_three, 0, 0], 0.)
        self.assertFalse(np.any(windows['row_index'] == 4))

    def test_tiny_cli_is_separate_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'split'
            split.mkdir()
            (split / 'train_regions.bed').write_text(
                ''.join(f'chr1\t{i*100}\t{(i+1)*100}\n' for i in range(8)))
            (split / 'test_regions.bed').write_text('chr1\t800\t1000\n')
            fields = ['bin_id', 'chrom', 'start', 'end', 'bin_size_bp',
                      'okseq', 'groseq', 'break_count']
            dataset = root / 'dataset.tsv'
            with dataset.open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(fields)
                for i in range(100):
                    writer.writerow([f'b{i}', 'chr1', i * 10, (i + 1) * 10,
                                     10, .1 * (i % 2), i % 3, i % 2])
            excluded = root / 'excluded.tsv'
            excluded.write_text('\t'.join(fields + ['exclusion_reason']) + '\n')
            output = root / 'context-result'
            argv = ['--dataset', str(dataset), '--excluded-bins', str(excluded),
                    '--split-dir', str(split), '--window-bins', '3',
                    '--boundary-buffer-bp', '0', '--width', '8', '--layers', '1',
                    '--heads', '2', '--feedforward', '16', '--dropout', '0',
                    '--epochs', '1', '--batch-size', '16', '--output-dir', str(output)]
            context.main(argv)
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual(config['status'], 'complete')
            self.assertEqual(config['architecture']['type'], 'context_transformer')
            self.assertTrue((output / 'model.pt').exists())
            self.assertTrue((output / 'break_probability.bedGraph').exists())
            with self.assertRaises(FileExistsError):
                context.main(argv)


if __name__ == '__main__':
    unittest.main()
