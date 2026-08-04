import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate.py"


class UnifiedEvaluateCliTest(unittest.TestCase):
    def _run(
        self,
        temporary: str,
        config: str,
        prediction: np.ndarray,
        target: np.ndarray,
        *extra: str,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        root = Path(temporary)
        prediction_root = root / "predictions"
        target_root = root / "targets"
        output_root = root / "output"
        prediction_root.mkdir()
        target_root.mkdir()
        np.save(prediction_root / "sample.npy", prediction)
        np.save(target_root / "sample.npy", target)
        command = [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ROOT / config),
            "--predictions",
            str(prediction_root),
            "--targets",
            str(target_root),
            "--output-dir",
            str(output_root),
            *extra,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
        self.assertTrue((output_root / "metrics.csv").is_file())
        return completed, result

    def test_segmentation_uses_dataset_confusion_and_ade_label_mapping(self):
        prediction = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        raw_ade_target = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            completed, result = self._run(
                temporary,
                "configs/segmentation/ade20k.yaml",
                prediction,
                raw_ade_target,
                "--target-ignore-value",
                "0",
                "--target-label-offset",
                "-1",
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual("dataset_confusion_matrix", result["aggregation"])
        self.assertAlmostEqual(1.0, result["metrics"]["miou"])
        self.assertEqual(3, result["metrics"]["valid_pixels"])

    def test_depth_outputs_weighted_raw_and_affine_metrics(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        prediction = 2.0 * target + 3.0
        with tempfile.TemporaryDirectory() as temporary:
            completed, result = self._run(
                temporary,
                "configs/depth/nyuv2.yaml",
                prediction,
                target,
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertGreater(result["metrics"]["raw"]["abs_rel"], 1.0)
        self.assertLess(result["metrics"]["affine_aligned"]["abs_rel"], 1e-6)
        self.assertEqual(4, result["metrics"]["raw"]["valid_pixels"])

    def test_normal_outputs_angular_metrics(self):
        target = np.array(
            [[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32
        )
        prediction = np.array(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed, result = self._run(
                temporary,
                "configs/normal/nyuv2.yaml",
                prediction,
                target,
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertAlmostEqual(45.0, result["metrics"]["mean_angular_error"], places=5)
        self.assertAlmostEqual(0.5, result["metrics"]["acc_11_25"])
        self.assertEqual(2, result["metrics"]["valid_pixels"])

    def _run_multi(
        self,
        temporary: str,
        config: str,
        samples: dict[str, tuple[np.ndarray, np.ndarray]],
        *extra: str,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        root = Path(temporary)
        prediction_root = root / "predictions"
        target_root = root / "targets"
        output_root = root / "output"
        prediction_root.mkdir()
        target_root.mkdir()
        for sample_id, (prediction, target) in samples.items():
            np.save(prediction_root / f"{sample_id}.npy", prediction)
            np.save(target_root / f"{sample_id}.npy", target)
        command = [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ROOT / config),
            "--predictions",
            str(prediction_root),
            "--targets",
            str(target_root),
            "--output-dir",
            str(output_root),
            *extra,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result_path = output_root / "metrics.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {}
        )
        return completed, result

    def test_out_of_range_segmentation_prediction_is_skipped_not_fatal(self):
        # ADE has 150 classes; class 200 on a GT-valid pixel is unscorable. The
        # good sample must still produce dataset metrics (M-2).
        good = (np.array([[0, 1], [2, 0]], dtype=np.uint8),
                np.array([[1, 2], [3, 0]], dtype=np.uint8))
        bad = (np.array([[200, 1], [2, 0]], dtype=np.uint8),
               np.array([[1, 2], [3, 0]], dtype=np.uint8))
        with tempfile.TemporaryDirectory() as temporary:
            completed, result = self._run_multi(
                temporary,
                "configs/segmentation/ade20k.yaml",
                {"good": good, "bad": bad},
                "--target-ignore-value",
                "0",
                "--target-label-offset",
                "-1",
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(["bad"], [entry["sample_id"] for entry in result["skipped"]])
        self.assertEqual(["good"], [entry["sample_id"] for entry in result["samples"]])
        self.assertAlmostEqual(1.0, result["metrics"]["miou"])

    def test_all_ignore_depth_target_is_skipped_not_fatal(self):
        # A target of all zeros has no pixel above min_depth => unscorable. The
        # remaining scorable sample must still aggregate (M-3), and the finite
        # metrics must serialize under allow_nan=False.
        good_target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        good = (2.0 * good_target + 3.0, good_target)
        bad = (np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as temporary:
            completed, result = self._run_multi(
                temporary,
                "configs/depth/nyuv2.yaml",
                {"good": good, "bad": bad},
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(["bad"], [entry["sample_id"] for entry in result["skipped"]])
        self.assertEqual(["good"], [entry["sample_id"] for entry in result["samples"]])
        self.assertEqual(4, result["metrics"]["raw"]["valid_pixels"])

    def test_mismatched_sample_ids_fail_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions"
            targets = root / "targets"
            predictions.mkdir()
            targets.mkdir()
            np.save(predictions / "a.npy", np.zeros((2, 2), dtype=np.uint8))
            np.save(targets / "b.npy", np.zeros((2, 2), dtype=np.uint8))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(ROOT / "configs/segmentation/ade20k.yaml"),
                    "--predictions",
                    str(predictions),
                    "--targets",
                    str(targets),
                    "--output-dir",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(2, completed.returncode)
        self.assertIn("sample IDs differ", completed.stderr)


if __name__ == "__main__":
    unittest.main()
