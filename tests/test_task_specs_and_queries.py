import tempfile
import unittest
from pathlib import Path

import numpy as np

from perception_diffusion.codecs import (
    DepthCodec,
    NormalCodec,
    SegmentationBinaryMaskCodec,
)
from perception_diffusion.data import load_ade20k_object_info
from perception_diffusion.inference import merge_query_scores
from perception_diffusion.task_specs import (
    build_segmentation_query_planner,
    build_task_specs,
)
from perception_diffusion.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class SegmentationVocabularyTest(unittest.TestCase):
    def test_ade20k_metadata_is_ordered_and_uses_canonical_synonym(self):
        content = (
            "Idx\tRatio\tTrain\tVal\tStuff\tName\n"
            "2\t0.2\t1\t1\t0\tbuilding, edifice\n"
            "1\t0.3\t1\t1\t1\twall\n"
            "3\t0.1\t1\t1\t1\tsky\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "objectInfo150.txt"
            path.write_text(content, encoding="utf-8")
            vocabulary = load_ade20k_object_info(path, expected_num_classes=3)

        self.assertEqual(("wall", "building", "sky"), vocabulary.class_names)
        self.assertEqual((0, 1, 2), vocabulary.class_ids)

    def test_smoke_query_planner_always_returns_full_dataset_vocabulary(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        spec = build_task_specs(config)["segmentation"]
        planner = build_segmentation_query_planner(spec)

        queries = planner.all_queries()
        self.assertEqual(3, len(queries))
        self.assertEqual([0, 1, 2], [query.class_id for query in queries])
        self.assertEqual(
            "segmentation mask of object_a",
            queries[1].prompt,
        )
        self.assertEqual((2, 1), tuple(len(chunk) for chunk in planner.chunks(2)))

        scores = np.array(
            [
                [[0.8, 0.1], [0.1, 0.1]],
                [[0.1, 0.9], [0.2, 0.1]],
                [[0.1, 0.0], [0.7, 0.8]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            np.array([[0, 1], [2, 2]], dtype=np.int64),
            merge_query_scores(scores, queries),
        )


class TaskSpecTest(unittest.TestCase):
    def test_multitask_registry_builds_all_task_boundaries(self):
        config = load_config(
            ROOT / "configs" / "multitask" / "stage1_shared_unet.yaml",
            expand_environment=False,
        )
        specs = build_task_specs(config)

        self.assertEqual({"segmentation", "depth", "normal"}, set(specs))
        self.assertIsInstance(
            specs["segmentation"].codec, SegmentationBinaryMaskCodec
        )
        self.assertIsInstance(specs["depth"].codec, DepthCodec)
        self.assertIsInstance(specs["normal"].codec, NormalCodec)


if __name__ == "__main__":
    unittest.main()
