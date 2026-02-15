import unittest

from core.circuit import CoreParams, build_core_circuit_with_syndrome
from utils.logging import (
    apply_x_correction,
    decode_logical_bit_majority,
    decode_repetition_syndrome_majority,
    parse_counts_key_default,
    score_logical_success,
    score_raw_majority_success,
)


class TestRepetitionCode(unittest.TestCase):
    def test_parse_counts_key_default_reverses_groups(self):
        # Qiskit prints registers MSB->LSB:
        # - data: d2d1d0
        # - synd: s1s0 for 1 cycle
        parsed = parse_counts_key_default("001 01", n_cycles=1)
        self.assertEqual(parsed.d_bits, "100")  # d0d1d2
        self.assertEqual(parsed.s_bits, "10")   # s0s1

    def test_decode_syndrome_majority_mapping(self):
        # s_bits is index order: s0 s1 for one cycle.
        self.assertIsNone(decode_repetition_syndrome_majority("00", n_cycles=1))
        self.assertEqual(decode_repetition_syndrome_majority("10", n_cycles=1), 0)
        self.assertEqual(decode_repetition_syndrome_majority("11", n_cycles=1), 1)
        self.assertEqual(decode_repetition_syndrome_majority("01", n_cycles=1), 2)

    def test_apply_x_correction(self):
        self.assertEqual(apply_x_correction("000", None), "000")
        self.assertEqual(apply_x_correction("000", 0), "100")
        self.assertEqual(apply_x_correction("000", 1), "010")
        self.assertEqual(apply_x_correction("000", 2), "001")

    def test_decode_logical_bit_majority(self):
        self.assertEqual(decode_logical_bit_majority("000"), "0")
        self.assertEqual(decode_logical_bit_majority("111"), "1")
        self.assertEqual(decode_logical_bit_majority("100"), "0")
        self.assertEqual(decode_logical_bit_majority("110"), "1")

    def test_score_success(self):
        # One shot pattern: data has a single flip on q0 ("001" printed -> d_bits "100"),
        # and syndrome indicates q0 flip ("01" printed -> s_bits "10").
        counts = {"001 01": 10}
        self.assertEqual(score_raw_majority_success(counts, n_cycles=1, ideal_logical="0"), 1.0)
        self.assertEqual(score_logical_success(counts, n_cycles=1, ideal_logical="0"), 1.0)

    def test_circuit_shapes(self):
        p = CoreParams(n_cycles=2, idle_ticks_data=0, idle_ticks_anc=0, logical_state="0")
        qc = build_core_circuit_with_syndrome(p, n_cycles=p.n_cycles, measure_data=True)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(qc.num_clbits, 3 + 2 * p.n_cycles)  # d(3) + s(2*n)


if __name__ == "__main__":
    unittest.main()

