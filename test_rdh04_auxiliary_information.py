import unittest

import numpy as np

from v1_method_for_tomoka import (
    bytes_to_bits,
    deserialize_auxiliary_information,
    get_reserved_lsb_mask,
    pack_location_map,
    serialize_auxiliary_information,
    unpack_location_map,
    write_lsb_bits,
)


class TestAuxiliaryInformation(unittest.TestCase):
    def test_location_map_pack_and_unpack_round_trip(self):
        location_map = np.zeros((5, 7), dtype=np.uint8)
        location_map[0, 0] = 1
        location_map[2, 3] = 2
        location_map[4, 6] = 1

        compressed = pack_location_map(location_map)
        recovered = unpack_location_map(compressed, location_map.shape)

        np.testing.assert_array_equal(recovered, location_map)

    def test_deserializes_embedded_header_and_location_map(self):
        shape = (256, 256)
        location_map = np.zeros(shape, dtype=np.uint8)
        location_map[0, 0] = 1
        location_map[255, 255] = 2
        compressed = pack_location_map(location_map)
        auxiliary = serialize_auxiliary_information(
            shape,
            stage1_stop_rank=123,
            stage2_stop_rank=456,
            payload_length=789,
            compressed_location_map=compressed,
        )
        bits = bytes_to_bits(auxiliary)
        coords, _ = get_reserved_lsb_mask(shape, len(bits))
        stego = write_lsb_bits(np.full(shape, 100, dtype=np.uint8), coords, bits)

        recovered = deserialize_auxiliary_information(stego)

        self.assertEqual(recovered["stage1_stop_rank"], 123)
        self.assertEqual(recovered["stage2_stop_rank"], 456)
        self.assertEqual(recovered["payload_length"], 789)
        self.assertEqual(recovered["auxiliary_bit_length"], len(bits))
        np.testing.assert_array_equal(recovered["location_map"], location_map)


if __name__ == "__main__":
    unittest.main()
