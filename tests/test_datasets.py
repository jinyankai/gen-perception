import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.io import savemat

from perception_diffusion.codecs import (
    DepthCodec,
    NormalCodec,
    SegmentationBinaryMaskCodec,
)
from perception_diffusion.data import (
    ADE20KDataset,
    CameraIntrinsics,
    NYUv2Dataset,
    build_dataloader,
    collate_task_samples,
    depth_to_normals,
    load_nyuv2_splits,
)
from perception_diffusion.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def write_ade_metadata(path: Path) -> None:
    rows = ["Idx\tRatio\tTrain\tVal\tName\tStuff"]
    rows.extend(
        f"{index}\t0\t0\t0\tclass_{index:03d}\t0" for index in range(1, 151)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class ADE20KDatasetTest(unittest.TestCase):
    @staticmethod
    def make_fixture(root: Path) -> None:
        image_root = root / "images" / "training"
        mask_root = root / "annotations" / "training"
        image_root.mkdir(parents=True)
        mask_root.mkdir(parents=True)
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[..., 0] = 127
        source_mask = np.array(
            [
                [0, 1, 1, 2, 2, 2],
                [0, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(image).save(image_root / "sample.jpg")
        Image.fromarray(source_mask).save(mask_root / "sample.png")
        write_ade_metadata(root / "objectInfo150.txt")

    def test_real_layout_label_remap_binary_query_and_collation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ADEChallengeData2016"
            self.make_fixture(root)

            dataset = ADE20KDataset(
                root,
                split="train",
                image_size=(4, 6),
                codec=SegmentationBinaryMaskCodec(),
                query_sampling="first_present",
                strict_protocol=False,
            )
            sample = dataset[0]
            self.assertEqual("segmentation", sample["task_name"])
            self.assertEqual(0, sample["query_class_id"])
            self.assertEqual((3, 4, 6), tuple(sample["target"].shape))
            self.assertEqual(torch.bool, sample["valid_mask"].dtype)
            self.assertEqual(255, int(sample["native_target"][0, 0, 0]))
            self.assertEqual(0, int(sample["native_target"][0, 0, 1]))
            self.assertEqual(1, int(sample["native_target"][0, 0, 3]))
            batch = collate_task_samples((sample, sample))
            self.assertEqual("segmentation", batch["task_name"])
            self.assertEqual((2, 3, 4, 6), tuple(batch["image"].shape))

    def test_load_native_labels_returns_unresized_mapped_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ADEChallengeData2016"
            self.make_fixture(root)

            # Request a model input size (8x12) that differs from the native
            # 4x6 mask so a resize would be observable if it leaked in.
            dataset = ADE20KDataset(
                root,
                split="train",
                image_size=(8, 12),
                codec=SegmentationBinaryMaskCodec(),
                query_sampling="first_present",
                strict_protocol=False,
            )
            native = dataset.load_native_labels(0)
            self.assertEqual((4, 6), native.shape)
            self.assertEqual(np.int64, native.dtype)
            expected = np.array(
                [
                    [255, 0, 0, 1, 1, 1],
                    [255, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
                dtype=np.int64,
            )
            np.testing.assert_array_equal(expected, native)
            # __getitem__'s native_target is resized to the model input size,
            # confirming load_native_labels bypasses that resize.
            self.assertEqual((1, 8, 12), tuple(dataset[0]["native_target"].shape))

    def test_config_driven_dataloader_builds_homogeneous_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ADEChallengeData2016"
            self.make_fixture(root)
            config = load_config(
                ROOT / "configs" / "segmentation" / "ade20k.yaml",
                expand_environment=False,
            )
            config["data"]["root"] = str(root)
            config["data"]["image_size"] = [4, 6]
            config["data"]["num_workers"] = 0
            loader = build_dataloader(
                config,
                "segmentation",
                batch_size=1,
                num_workers=0,
                shuffle=False,
                strict_protocol=False,
            )
            batch = next(iter(loader))
            self.assertEqual("segmentation", batch["task_name"])
            self.assertEqual((1, 3, 4, 6), tuple(batch["target"].shape))


class NYUv2GeometryTest(unittest.TestCase):
    def test_fronto_parallel_plane_normals_face_camera(self):
        intrinsics = CameraIntrinsics(
            fx=4.0, fy=4.0, cx=2.5, cy=1.5, width=6, height=4
        )
        depth = np.full((4, 6), 2.0, dtype=np.float32)
        normals, valid = depth_to_normals(
            depth,
            intrinsics=intrinsics,
            max_relative_depth_jump=0.1,
        )
        self.assertTrue(valid[1:-1, 1:-1].all())
        np.testing.assert_allclose(normals[0, valid], 0.0, atol=1e-6)
        np.testing.assert_allclose(normals[1, valid], 0.0, atol=1e-6)
        np.testing.assert_allclose(normals[2, valid], -1.0, atol=1e-6)

    def test_depth_discontinuity_is_invalidated(self):
        intrinsics = CameraIntrinsics(
            fx=4.0, fy=4.0, cx=2.5, cy=1.5, width=6, height=4
        )
        depth = np.ones((4, 6), dtype=np.float32)
        depth[:, 3:] = 3.0
        _, valid = depth_to_normals(
            depth,
            intrinsics=intrinsics,
            max_relative_depth_jump=0.05,
        )
        self.assertFalse(valid[:, 2:4].any())


class NYUv2DatasetTest(unittest.TestCase):
    def make_raw_fixture(self, root: Path) -> None:
        sample_count, height, width = 5, 4, 6
        with h5py.File(root / "nyu_depth_v2_labeled.mat", "w") as handle:
            images = handle.create_dataset(
                "images", shape=(sample_count, 3, width, height), dtype=np.uint8
            )
            depths = handle.create_dataset(
                "depths", shape=(sample_count, width, height), dtype=np.float32
            )
            for index in range(sample_count):
                image = np.full((height, width, 3), 20 + index, dtype=np.uint8)
                depth = np.full((height, width), 1.0 + index, dtype=np.float32)
                images[index] = np.transpose(image, (2, 1, 0))
                depths[index] = depth.T
        savemat(
            root / "splits.mat",
            {
                "trainNdxs": np.array([[1, 2, 3]], dtype=np.int32),
                "testNdxs": np.array([[4, 5]], dtype=np.int32),
            },
        )

    def test_official_mat_axis_order_split_and_depth_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_raw_fixture(root)
            splits = load_nyuv2_splits(
                root / "splits.mat", sample_count=5, strict_protocol=False
            )
            np.testing.assert_array_equal(splits["train"], [0, 1, 2])
            dataset = NYUv2Dataset(
                root,
                split="train",
                task="depth",
                image_size=(4, 6),
                codec=DepthCodec(0.1, 10.0),
                source="raw",
                strict_protocol=False,
                native_size=(4, 6),
            )
            sample = dataset[1]
            self.assertEqual("nyuv2/train/0002", sample["sample_id"])
            np.testing.assert_allclose(sample["native_target"].numpy(), 2.0)
            self.assertTrue(bool(sample["valid_mask"].all()))
            dataset.close()

    def test_processed_normal_sample_loads_without_raw_mat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            (processed / "test").mkdir(parents=True)
            normal = np.zeros((3, 4, 6), dtype=np.float32)
            normal[2] = -1.0
            np.savez_compressed(
                processed / "test" / "0001.npz",
                image=np.zeros((4, 6, 3), dtype=np.uint8),
                depth=np.ones((4, 6), dtype=np.float32),
                normal=normal,
                normal_valid=np.ones((4, 6), dtype=bool),
            )
            (processed / "manifest.json").write_text(
                '{"format":"gen-perception-nyuv2-v1","splits":{"test":[0]}}',
                encoding="utf-8",
            )
            dataset = NYUv2Dataset(
                root,
                split="test",
                task="normal",
                image_size=(4, 6),
                codec=NormalCodec(),
                source="processed",
                strict_protocol=False,
                native_size=(4, 6),
            )
            sample = dataset[0]
            norms = torch.linalg.vector_norm(sample["native_target"], dim=0)
            torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_horizontal_flip_reflects_normal_x_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            (processed / "test").mkdir(parents=True)
            normal = np.zeros((3, 4, 6), dtype=np.float32)
            normal[0] = 0.6
            normal[2] = -0.8
            np.savez_compressed(
                processed / "test" / "0001.npz",
                image=np.zeros((4, 6, 3), dtype=np.uint8),
                depth=np.ones((4, 6), dtype=np.float32),
                normal=normal,
                normal_valid=np.ones((4, 6), dtype=bool),
            )
            (processed / "manifest.json").write_text(
                '{"format":"gen-perception-nyuv2-v1","splits":{"test":[0]}}',
                encoding="utf-8",
            )
            dataset = NYUv2Dataset(
                root,
                split="test",
                task="normal",
                image_size=(4, 6),
                codec=NormalCodec(),
                source="processed",
                horizontal_flip_probability=1.0,
                strict_protocol=False,
                native_size=(4, 6),
            )
            sample = dataset[0]
            torch.testing.assert_close(
                sample["native_target"][0], torch.full((4, 6), -0.6)
            )
            torch.testing.assert_close(
                sample["native_target"][2], torch.full((4, 6), -0.8)
            )


if __name__ == "__main__":
    unittest.main()
