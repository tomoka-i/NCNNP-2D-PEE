import unittest

import numpy as np
import torch

from v1_method_for_tomoka import (
    embed_stage_2dpee,
    embed_two_stage_2dpee,
    extract_stage_2dpee,
    extract_two_stage_2dpee,
    get_cross_complexity_matrix,
    predict_target_parity,
)


class ConstantPredictor(torch.nn.Module):
    def forward(self, image):
        return image * 0 + (100.0 / 255.0)


class TestBlindCrossComplexity(unittest.TestCase):
    def setUp(self):
        self.model = ConstantPredictor().eval()
        self.device = "cpu"
        self.payload = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]

        # Use a signed type while constructing negative prediction errors.
        original = np.full((8, 8), 100, dtype=np.int16)
        y_idx, x_idx = np.indices(original.shape)
        self.star_mask = (y_idx + x_idx) % 2 == 0
        self.dot_mask = ~self.star_mask
        star_coords = np.argwhere(self.star_mask)
        dot_coords = np.argwhere(self.dot_mask)

        # Stage 2 (star) contains C-type pairs and carries the payload.
        for index, (y, x) in enumerate(star_coords):
            original[y, x] += (0, 2)[index % 2]

        # Stage 1 (dot) contains only D-type pairs.  It shifts dot pixels,
        # so Stage 2 must use the Stage-1 stego dot complexity matrix.
        dot_pair_errors = [(3, -3), (2, 4), (-2, 3), (-4, -2)]
        for pair_index in range(len(dot_coords) // 2):
            error1, error2 = dot_pair_errors[pair_index % len(dot_pair_errors)]
            y1, x1 = dot_coords[pair_index * 2]
            y2, x2 = dot_coords[pair_index * 2 + 1]
            original[y1, x1] += error1
            original[y2, x2] += error2

        self.original = original.astype(np.uint8)

    def test_encoder_and_decoder_reproduce_cross_complexity(self):
        prediction1 = predict_target_parity(self.model, self.original, 1, self.device)
        complexity1_encoder = get_cross_complexity_matrix(self.original, 1)
        stego1, used1, stop_rank1 = embed_stage_2dpee(
            self.original,
            prediction1,
            1,
            self.payload,
            complexity1_encoder,
            target_ec=len(self.payload),
        )

        stego1_array = np.asarray(stego1)
        prediction2_encoder = predict_target_parity(
            self.model, stego1_array, 0, self.device
        )
        complexity2_encoder = get_cross_complexity_matrix(stego1_array, 0)
        stego2, used_total, stop_rank2 = embed_stage_2dpee(
            stego1_array,
            prediction2_encoder,
            0,
            self.payload,
            complexity2_encoder,
            bit_ptr_start=used1,
            target_ec=len(self.payload),
        )

        stego2_array = np.asarray(stego2)
        prediction2_decoder = predict_target_parity(
            self.model, stego2_array, 0, self.device
        )
        complexity2_decoder = get_cross_complexity_matrix(stego2_array, 0)
        np.testing.assert_array_equal(prediction2_encoder, prediction2_decoder)
        np.testing.assert_allclose(
            complexity2_encoder[self.star_mask],
            complexity2_decoder[self.star_mask],
        )

        recovered_stage1, stage2_bits = extract_stage_2dpee(
            stego2_array,
            prediction2_decoder,
            0,
            complexity2_decoder,
            stop_rank2,
        )

        recovered_stage1_array = np.asarray(recovered_stage1)
        prediction1_decoder = predict_target_parity(
            self.model, recovered_stage1_array, 1, self.device
        )
        complexity1_decoder = get_cross_complexity_matrix(recovered_stage1_array, 1)
        np.testing.assert_array_equal(prediction1, prediction1_decoder)
        np.testing.assert_allclose(
            complexity1_encoder[self.dot_mask],
            complexity1_decoder[self.dot_mask],
        )

        recovered, stage1_bits = extract_stage_2dpee(
            recovered_stage1_array,
            prediction1_decoder,
            1,
            complexity1_decoder,
            stop_rank1,
        )

        self.assertEqual(used1, 0)
        self.assertEqual(used_total, len(self.payload))
        self.assertEqual(stage1_bits + stage2_bits, self.payload)
        np.testing.assert_array_equal(np.asarray(recovered), self.original)

    def test_public_api_recovers_payload_and_cover_image(self):
        # RDH-04 stores auxiliary information in border LSBs, so this public
        # end-to-end test needs a border large enough to hold the header.
        original = np.full((256, 256), 100, dtype=np.uint8)
        stego, info = embed_two_stage_2dpee(
            self.model,
            original,
            self.payload,
            self.device,
            target_ec=len(self.payload),
        )
        recovered, extracted = extract_two_stage_2dpee(
            self.model,
            np.asarray(stego),
            self.device,
        )

        self.assertEqual(extracted, self.payload)
        np.testing.assert_array_equal(np.asarray(recovered), original)


if __name__ == "__main__":
    unittest.main()
