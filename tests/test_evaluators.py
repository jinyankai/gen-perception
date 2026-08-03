import unittest

import numpy as np

from perception_diffusion.evaluation import DepthEvaluator, NormalEvaluator, SegmentationEvaluator


class SegmentationEvaluatorTest(unittest.TestCase):
    def test_perfect_prediction_ignores_ignore_label(self):
        target = np.array([[0, 1, 255], [2, 1, 0]], dtype=np.int64)
        result = SegmentationEvaluator(3, ignore_label=255).evaluate(target, target)
        self.assertAlmostEqual(1.0, result["miou"])
        self.assertAlmostEqual(1.0, result["pixel_accuracy"])
        self.assertEqual(5, result["valid_pixels"])

    def test_invalid_prediction_raises(self):
        target = np.array([[0, 1]], dtype=np.int64)
        prediction = np.array([[0, 7]], dtype=np.int64)
        with self.assertRaises(ValueError):
            SegmentationEvaluator(2).evaluate(prediction, target)


class DepthEvaluatorTest(unittest.TestCase):
    def test_affine_alignment_recovers_target(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        prediction = 2.0 * target + 3.0
        result = DepthEvaluator(affine_align=True).evaluate(prediction, target)
        self.assertGreater(result["raw"]["abs_rel"], 1.0)
        self.assertLess(result["affine_aligned"]["abs_rel"], 1e-6)
        self.assertAlmostEqual(0.5, result["alignment"]["scale"], places=6)
        self.assertAlmostEqual(-1.5, result["alignment"]["shift"], places=6)

    def test_invalid_target_pixels_are_excluded(self):
        target = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
        prediction = np.ones_like(target)
        result = DepthEvaluator(affine_align=False).evaluate(prediction, target)
        self.assertEqual(2, result["raw"]["valid_pixels"])

    def test_constant_prediction_aligns_to_target_mean(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        prediction = np.ones_like(target)
        result = DepthEvaluator(affine_align=True).evaluate(prediction, target)
        self.assertAlmostEqual(0.0, result["alignment"]["scale"])
        self.assertAlmostEqual(2.5, result["alignment"]["shift"])


class NormalEvaluatorTest(unittest.TestCase):
    def test_known_angles(self):
        target = np.array([[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)
        prediction = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32)
        result = NormalEvaluator().evaluate(prediction, target)
        self.assertAlmostEqual(45.0, result["mean_angular_error"], places=5)
        self.assertAlmostEqual(45.0, result["median_angular_error"], places=5)
        self.assertAlmostEqual(0.5, result["acc_11_25"])


if __name__ == "__main__":
    unittest.main()
