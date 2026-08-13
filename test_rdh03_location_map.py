import unittest

import numpy as np
import torch

from v1_method_for_tomoka import (
    LOCATION_MAP_FROM_255,
    LOCATION_MAP_FROM_ZERO,
    LOCATION_MAP_UNCHANGED,
    embed_two_stage_2dpee,
    extract_two_stage_2dpee,
    preprocess_boundary_pixels,
    restore_boundary_pixels,
)


class ConstantPredictor(torch.nn.Module):
    def forward(self, image):
        return image * 0 + (100.0 / 255.0)


class TestLocationMap(unittest.TestCase):
    def test_preprocess_and_restore_boundary_pixels(self):
        original = np.array(
            [[0, 1, 127], [254, 255, 100]],
            dtype=np.uint8,
        )

        prepared, location_map = preprocess_boundary_pixels(original)

        expected_prepared = np.array(
            [[1, 1, 127], [254, 254, 100]],
            dtype=np.uint8,
        )
        expected_map = np.array(
            [
                [LOCATION_MAP_FROM_ZERO, LOCATION_MAP_UNCHANGED, LOCATION_MAP_UNCHANGED],
                [LOCATION_MAP_UNCHANGED, LOCATION_MAP_FROM_255, LOCATION_MAP_UNCHANGED],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(prepared, expected_prepared)
        np.testing.assert_array_equal(location_map, expected_map)
        np.testing.assert_array_equal(
            restore_boundary_pixels(prepared, location_map),
            original,
        )

    def test_two_stage_round_trip_restores_boundary_pixels(self):
        model = ConstantPredictor().eval()
        original = np.full((8, 8), 100, dtype=np.uint8)
        original[0, 0] = 0
        original[0, 1] = 255
        original[7, 6] = 0
        original[7, 7] = 255
        payload = [1, 0, 1, 1]

        stego, info = embed_two_stage_2dpee(
            model,
            original,
            payload,
            "cpu",
            target_ec=len(payload),
        )
        recovered, extracted = extract_two_stage_2dpee(
            model,
            np.asarray(stego),
            "cpu",
            info["stage1_stop_rank"],
            info["stage2_stop_rank"],
            info["payload_length"],
            location_map=info["location_map"],
        )

        self.assertEqual(extracted, payload)
        np.testing.assert_array_equal(np.asarray(recovered), original)


if __name__ == "__main__":
    unittest.main()
