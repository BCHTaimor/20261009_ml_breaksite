"""Contract tests for the shared-data loader and break-count model.

Contract targets: ml_data.load_regions/classify_bin/load_datasets;
train_model.Preprocessor/BreakCountNet and training CLI.

Partitions: adjacent same-label regions; mixed-resolution bins; chromosome
ends versus internal boundaries; boundary-crossing and buffered bins;
training-only normalization; raw-count validation; synthetic complete run.
Depth partitions: minimum 1, default 5, requested 30; reject nonpositive/noninteger
depth; 30-layer CLI checkpoint records actual architecture and reloads.
Invariants: no genomic overlap between partitions across resolutions;
held-out values cannot alter fitted preprocessing; positive count output;
saved runs reuse shared inputs and never duplicate dataset tables.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ml_data import classify_bin, load_datasets, load_regions
from train_model import BreakCountNet, Preprocessor


COLUMNS = ["bin_id", "chrom", "start", "end", "bin_size_bp", "okseq", "groseq", "break_count"]


def write_table(path, rows):
    """Write a minimal shared dataset fixture using the public TSV schema."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def bin_row(start, size, **updates):
    """Return one valid aligned synthetic bin with raw-count labels."""
    row = dict(zip(COLUMNS, [f"hg19:chr1:{start}-{start + size}", "chr1", start,
                            start + size, size, 0.25, 3.0, 2]))
    row.update(updates)
    return row


class RegionSelectionTests(unittest.TestCase):
    def test_half_open_containment_and_internal_buffer(self):
        regions = [("chr1", 0, 20000, "train"), ("chr1", 20000, 40000, "test")]
        self.assertEqual(classify_bin("chr1", 15000, 20000, regions, 0), "train")
        self.assertEqual(classify_bin("chr1", 20000, 25000, regions, 0), "test")
        self.assertEqual(classify_bin("chr1", 15000, 25000, regions, 0), "excluded")
        self.assertEqual(classify_bin("chr1", 15000, 20000, regions, 1000), "excluded")
        self.assertEqual(classify_bin("chr1", 20000, 25000, regions, 1000), "excluded")
        self.assertEqual(classify_bin("chr1", 0, 5000, regions, 1000), "train")
        self.assertEqual(classify_bin("chr1", 35000, 40000, regions, 1000), "test")

    def test_adjacent_training_rows_merge_before_bin_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            (split / "train_regions.bed").write_text(
                "".join(f"chr1\t{i}\t{i + 10000}\n" for i in range(0, 40000, 10000)))
            (split / "test_regions.bed").write_text("chr1\t40000\t50000\n")
            regions = load_regions(split, 0.2, 19)
            merged = [r for r in regions if r[3] == "train" and r[2] - r[1] >= 20000]
            self.assertTrue(merged)
            start = merged[0][1]
            self.assertEqual(classify_bin("chr1", start + 5000, start + 15000, regions, 0), "train")

    def test_validation_selection_keeps_mixed_resolutions_in_same_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            (split / "train_regions.bed").write_text(
                "".join(f"chr1\t{i}\t{i + 20000}\n" for i in range(0, 100000, 20000)))
            (split / "test_regions.bed").write_text("chr1\t100000\t120000\n")
            regions = load_regions(split, 0.2, 19)
            self.assertEqual(regions, load_regions(split, 0.2, 19))
            observed = set()
            for start in range(0, 120000, 10000):
                coarse = classify_bin("chr1", start, start + 10000, regions, 0)
                fine_a = classify_bin("chr1", start, start + 5000, regions, 0)
                fine_b = classify_bin("chr1", start + 5000, start + 10000, regions, 0)
                self.assertEqual(coarse, fine_a)
                self.assertEqual(coarse, fine_b)
                observed.add(coarse)
            self.assertEqual(observed, {"train", "validation", "test"})


class DatasetValidationTests(unittest.TestCase):
    def test_multiple_resolutions_include_explicit_excluded_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "5kb.tsv", Path(directory) / "10kb.tsv"
            write_table(a, [bin_row(0, 5000), bin_row(15000, 5000)])
            write_table(b, [bin_row(0, 10000), bin_row(20000, 10000)])
            regions = [("chr1", 0, 20000, "train"), ("chr1", 20000, 40000, "test")]
            data = load_datasets([a, b], regions, 1000)
            np.testing.assert_array_equal(data["x"][:, 2], [5000, 5000, 10000, 10000])
            self.assertEqual(list(data["partition"]), ["train", "excluded", "train", "excluded"])
            np.testing.assert_array_equal(data["y"], [2, 2, 2, 2])

    def test_invalid_measurements_and_fractional_labels_rejected(self):
        regions = [("chr1", 0, 20000, "train")]
        for update in [{"groseq": ""}, {"okseq": "nan"}, {"break_count": 1.5}, {"break_count": -1}]:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "dataset.tsv"
                write_table(path, [bin_row(0, 5000, **update)])
                with self.assertRaises(ValueError):
                    load_datasets([path], regions, 0)


class ModelContractTests(unittest.TestCase):
    def test_configurable_depth_preserves_count_output_and_reload(self):
        for depth in (1, 30):
            with self.subTest(depth=depth):
                torch.manual_seed(11)
                model = BreakCountNet(depth)
                self.assertEqual(sum(isinstance(x, torch.nn.ReLU) for x in model.modules()), depth)
                expected = [3, 64, 1] if depth == 1 else [3] + [64] * 27 + [32, 16, 8, 1]
                self.assertEqual(model.architecture, expected)
                prediction = model(torch.zeros(2, 3))
                self.assertEqual(tuple(prediction.shape), (2,))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertTrue((prediction > 0).all())
                restored = BreakCountNet(depth)
                restored.load_state_dict(model.state_dict())
                self.assertTrue(torch.equal(prediction, restored(torch.zeros(2, 3))))

    def test_invalid_hidden_layer_count_is_rejected(self):
        for depth in (0, -1, 1.5, True):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, 'positive integer'):
                BreakCountNet(depth)

    def test_requested_architecture_produces_positive_expected_counts(self):
        model = BreakCountNet()
        layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(x.in_features, x.out_features) for x in layers],
                         [(3, 64), (64, 64), (64, 32), (32, 16), (16, 8), (8, 1)])
        prediction = model(torch.tensor([[0., 0., 0.], [-3., 5., 2.]]))
        self.assertEqual(prediction.numel(), 2)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_preprocessing_is_fitted_only_on_training_data(self):
        train = np.array([[-1., 0., 5000.], [1., 3., 10000.]])
        prep = Preprocessor().fit(train)
        before = prep.transform(train).copy()
        np.testing.assert_allclose(before, [[-1, -1, -1], [1, 1, 1]], atol=1e-6)
        held_out = prep.transform(np.array([[1., 1e9, 100000.]]))
        self.assertTrue(np.isfinite(held_out).all())
        np.testing.assert_array_equal(prep.transform(train), before)


class TrainingIntegrationTests(unittest.TestCase):
    def test_synthetic_cli_saves_reloadable_model_without_dataset_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split"
            split.mkdir()
            (split / "manifest.json").write_text(json.dumps({"seed": 17, "genome_build": "hg19"}))
            (split / "train_regions.bed").write_text(
                "".join(f"chr1\t{i}\t{i + 20000}\n" for i in range(0, 100000, 20000)))
            (split / "test_regions.bed").write_text("chr1\t100000\t120000\n")
            paths = [root / "5kb.tsv", root / "10kb.tsv"]
            for path, size in zip(paths, [5000, 10000]):
                write_table(path, [bin_row(start, size, okseq=(start % 20000) / 20000,
                                          groseq=start / 5000, break_count=(start // 5000) % 4)
                                   for start in range(0, 120000, size)])
            original_inputs = [p.read_bytes() for p in paths]
            output = root / "run"
            command = [sys.executable, str(ROOT / "scripts" / "train_model.py"),
                       "--dataset", str(paths[0]), "--dataset", str(paths[1]),
                       "--split-dir", str(split), "--boundary-buffer-bp", "0",
                       "--target-kind", "raw-count", "--output-dir", str(output),
                       "--epochs", "2", "--patience", "2", "--batch-size", "8",
                       "--model-seed", "11", "--validation-seed", "19"]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for relative in ["config.json", "metrics.tsv", "history.tsv", "validation_regions.bed", "model/model.pt"]:
                self.assertTrue((output / relative).is_file(), relative)
            checkpoint = torch.load(output / "model" / "model.pt", map_location="cpu", weights_only=True)
            model = BreakCountNet()
            model.load_state_dict(checkpoint["state_dict"])
            self.assertEqual(checkpoint["architecture"], [3, 64, 64, 32, 16, 8, 1])
            self.assertEqual(len(checkpoint["preprocessor"]["mean"]), 3)
            regions = load_regions(split, 0.15, 19)
            loaded = load_datasets(paths, regions, 0)
            training = loaded["x"][loaded["partition"] == "train"].copy()
            training[:, 1] = np.log1p(training[:, 1])
            training[:, 2] = np.log(training[:, 2])
            np.testing.assert_allclose(checkpoint["preprocessor"]["mean"], training.mean(axis=0))
            self.assertTrue((model(torch.zeros(2, 3)) > 0).all())
            self.assertEqual([p.read_bytes() for p in paths], original_inputs)
            self.assertFalse(list(output.rglob("dataset.tsv")))
            self.assertFalse((output / "predictions.tsv").exists())
            reproduced = root / "reproduced"
            repeat_command = command.copy()
            repeat_command[repeat_command.index("--output-dir") + 1] = str(reproduced)
            repeated = subprocess.run(repeat_command, text=True, capture_output=True, timeout=90)
            self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
            repeated_checkpoint = torch.load(reproduced / "model" / "model.pt",
                                             map_location="cpu", weights_only=True)
            for name, parameter in checkpoint["state_dict"].items():
                self.assertTrue(torch.equal(parameter, repeated_checkpoint["state_dict"][name]), name)
            self.assertEqual((output / "metrics.tsv").read_bytes(),
                             (reproduced / "metrics.tsv").read_bytes())
            deep_output = root / "deep"
            deep_command = command.copy()
            deep_command[deep_command.index("--output-dir") + 1] = str(deep_output)
            deep = subprocess.run(deep_command + ["--hidden-layers", "30"],
                                  text=True, capture_output=True, timeout=90)
            self.assertEqual(deep.returncode, 0, deep.stdout + deep.stderr)
            deep_config = json.loads((deep_output / "config.json").read_text())
            deep_checkpoint = torch.load(deep_output / "model/model.pt", weights_only=True)
            deep_model = BreakCountNet(30)
            deep_model.load_state_dict(deep_checkpoint["state_dict"])
            self.assertEqual(deep_config["arguments"]["hidden_layers"], 30)
            self.assertEqual(deep_config["architecture"], deep_model.architecture)
            self.assertEqual(deep_checkpoint["architecture"], deep_model.architecture)
            rerun = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(rerun.returncode, 0, "Existing run must not be overwritten")


if __name__ == "__main__":
    unittest.main()
