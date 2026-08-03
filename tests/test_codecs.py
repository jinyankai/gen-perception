import unittest

import numpy as np

from perception_diffusion.codecs import (
    DepthCodec,
    NormalCodec,
    SegmentationBinaryMaskCodec,
    SegmentationIdCodec,
    SegmentationPaletteCodec,
)


class SegmentationCodecTest(unittest.TestCase):
    def test_binary_query_mask_round_trip_uses_vae_range(self):
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        codec = SegmentationBinaryMaskCodec()
        encoded = codec.encode(mask)

        self.assertEqual(-1.0, float(encoded.values.min()))
        self.assertEqual(1.0, float(encoded.values.max()))
        np.testing.assert_array_equal(mask, codec.decode(encoded.values))

    def setUp(self):
        self.labels = np.array([[0, 1, 255], [2, 1, 0]], dtype=np.int64)

    def test_palette_round_trip_preserves_ids_and_ignore(self):
        codec = SegmentationPaletteCodec(num_classes=3, ignore_label=255)
        encoded = codec.encode(self.labels)
        decoded = codec.decode(encoded.values, encoded.valid_mask)
        np.testing.assert_array_equal(decoded, self.labels)
        self.assertEqual((3, 2, 3), encoded.values.shape)
        self.assertTrue(np.all(encoded.values <= 1.0))
        self.assertTrue(np.all(encoded.values >= -1.0))

    def test_id_round_trip_is_available_for_diagnostics(self):
        codec = SegmentationIdCodec(num_classes=3, ignore_label=255)
        encoded = codec.encode(self.labels)
        decoded = codec.decode(encoded.values, encoded.valid_mask)
        np.testing.assert_array_equal(decoded, self.labels)

    def test_invalid_class_raises(self):
        codec = SegmentationPaletteCodec(num_classes=3)
        with self.assertRaises(ValueError):
            codec.encode(np.array([[0, 3]], dtype=np.int64))


class DepthCodecTest(unittest.TestCase):
    def test_all_representations_round_trip_valid_depth(self):
        depth = np.array([[0.1, 1.0], [5.0, np.nan]], dtype=np.float32)
        for representation in ("linear", "inverse", "log"):
            with self.subTest(representation=representation):
                codec = DepthCodec(0.1, 10.0, representation)
                encoded = codec.encode(depth)
                decoded = codec.decode(encoded.values, encoded.valid_mask)
                np.testing.assert_allclose(
                    decoded[encoded.valid_mask],
                    depth[encoded.valid_mask],
                    rtol=1e-5,
                )
                self.assertTrue(np.isnan(decoded[1, 1]))


class NormalCodecTest(unittest.TestCase):
    def test_round_trip_renormalizes(self):
        normals = np.array(
            [[[2.0, 0.0], [0.0, 0.0]], [[0.0, 3.0], [0.0, 0.0]], [[0.0, 0.0], [4.0, 0.0]]],
            dtype=np.float32,
        )
        codec = NormalCodec()
        encoded = codec.encode(normals)
        decoded = codec.decode(encoded.values, encoded.valid_mask)
        np.testing.assert_allclose(
            np.linalg.norm(decoded[:, encoded.valid_mask], axis=0), 1.0, atol=1e-6
        )
        np.testing.assert_array_equal(decoded[:, 1, 1], np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
