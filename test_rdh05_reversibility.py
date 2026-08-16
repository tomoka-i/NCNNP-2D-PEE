import unittest

import numpy as np
import torch

from v1_method_for_tomoka import (
    embed_two_stage_2dpee,
    extract_two_stage_2dpee,
)


class ConstantPredictor(torch.nn.Module):
    """Deterministic predictor for reversible-pipeline tests."""

    def forward(self, image):
        return image * 0 + (100.0 / 255.0)


class TestEndToEndReversibility(unittest.TestCase):
    def setUp(self):
        self.model = ConstantPredictor().eval()
        self.device = "cpu"

    @staticmethod
    def make_payload(length):
        return [((index * 7) + 3) % 2 for index in range(length)]

    def assert_round_trip(self, original, payload):
        stego, info = embed_two_stage_2dpee(
            self.model,
            original,
            payload,
            self.device,
            target_ec=len(payload),
        )
        recovered, extracted = extract_two_stage_2dpee(
            self.model,
            np.asarray(stego),
            self.device,
        )

        self.assertEqual(info["payload_length"], len(payload))
        self.assertEqual(extracted, payload)
        np.testing.assert_array_equal(np.asarray(recovered), original)
        return np.asarray(stego)

    def test_round_trip_preserves_reserved_lsb_values_and_payload_lengths(self):
        original = np.full((256, 256), 100, dtype=np.uint8)
        original[0, :] = 101
        original[-1, :] = 101
        original[:, 0] = 101
        original[:, -1] = 101

        for length in (0, 1, 257):
            with self.subTest(payload_length=length):
                self.assert_round_trip(original, self.make_payload(length))

    def test_round_trip_handles_zero_and_255_at_border_and_interior(self):
        original = np.full((256, 256), 100, dtype=np.uint8)
        original[0, 0] = 0
        original[0, -1] = 255
        original[-1, 0] = 255
        original[-1, -1] = 0
        original[32, 64] = 0
        original[192, 160] = 255

        self.assert_round_trip(original, self.make_payload(129))

    def test_rejects_payload_when_pee_capacity_cannot_cover_auxiliary_data(self):
        # All errors are far outside the embedding range for the constant
        # predictor, so no PEE payload capacity exists after reservation.
        low_capacity_image = np.full((256, 256), 1, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "insufficient PEE capacity"):
            embed_two_stage_2dpee(
                self.model,
                low_capacity_image,
                self.make_payload(1),
                self.device,
                target_ec=1,
            )

    def test_rejects_tampered_auxiliary_header(self):
        original = np.full((256, 256), 100, dtype=np.uint8)
        stego = self.assert_round_trip(original, self.make_payload(32))
        tampered = stego.copy()
        tampered[0, 0] ^= 1

        with self.assertRaisesRegex(
            ValueError,
            "auxiliary-information header is missing or unsupported",
        ):
            extract_two_stage_2dpee(self.model, tampered, self.device)


if __name__ == "__main__":
    unittest.main()
