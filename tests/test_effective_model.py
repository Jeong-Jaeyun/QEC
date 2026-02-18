import math
import unittest
import random
from dataclasses import replace

import numpy as np

from core.effective_model import (
    MODE_AUTO,
    MODE_FULL,
    MODE_NAIVE,
    EffectiveModelParams,
    apply_reset_fault,
    bayes_filter_step,
    build_full_transition_matrix,
    build_decoder_transition_matrix,
    decode_map_correction_target,
    estimate_logical_dephasing_rate,
    estimate_logical_dephasing_rate_with_ci,
    effective_observation_error,
    emission_probability,
    is_sparse_measurement_round,
    sample_ideal_syndrome_from_rho,
    sample_monitoring_observation_from_state,
    sample_noisy_syndrome_from_rho,
    sparse_measurement_count,
    sparse_measurement_rounds,
    run_monitoring_filter,
    run_three_baselines,
    simulate_lifetime,
    syndrome_for_state,
    sweep_k,
    total_time_for_rounds,
)


class TestEffectiveModel(unittest.TestCase):
    def test_syndrome_mapping(self):
        self.assertEqual(syndrome_for_state(0), (0, 0))
        self.assertEqual(syndrome_for_state(1), (1, 0))
        self.assertEqual(syndrome_for_state(2), (1, 1))
        self.assertEqual(syndrome_for_state(3), (0, 1))

    def test_transition_matrix_rows_sum_to_one(self):
        p = EffectiveModelParams(gamma_x=0.04, chi=3.0, zeta=0.05, p_m=0.02, p_r=0.02, shots=50, max_rounds=20)
        mat = build_full_transition_matrix(p, MODE_AUTO)
        self.assertEqual(len(mat), 5)
        for row in mat:
            self.assertAlmostEqual(sum(row), 1.0, places=10)

    def test_three_baselines_and_k_sweep(self):
        p = EffectiveModelParams(
            gamma_x=0.03,
            chi=4.0,
            zeta=0.04,
            eta_t=1.0,
            p_m=0.02,
            p_r=0.02,
            k=5,
            shots=160,
            max_rounds=80,
            seed=19,
        )
        bundle = run_three_baselines(p, k_sparse=5)
        self.assertEqual(bundle.full.k, 1)
        self.assertEqual(bundle.naive.k, 5)
        self.assertEqual(bundle.autonomous.k, 5)
        self.assertGreaterEqual(bundle.full.logical_failure_prob, 0.0)
        self.assertGreaterEqual(bundle.naive.logical_failure_prob, 0.0)
        self.assertGreaterEqual(bundle.autonomous.logical_failure_prob, 0.0)

        res = sweep_k(p, MODE_AUTO, k_values=[2, 3, 4, 5, 6], full_stats=bundle.full)
        self.assertIn(res.k_star, [2, 3, 4, 5, 6])
        self.assertEqual(len(res.points), 5)

    def test_simulate_lifetime_basic(self):
        p = EffectiveModelParams(shots=120, max_rounds=60, seed=7)
        stats_full = simulate_lifetime(p, MODE_FULL)
        stats_naive = simulate_lifetime(p, MODE_NAIVE)
        self.assertGreater(stats_full.lifetime_time, 0.0)
        self.assertGreater(stats_naive.lifetime_time, 0.0)
        self.assertGreaterEqual(stats_full.survival_prob, 0.0)
        self.assertLessEqual(stats_full.survival_prob, 1.0)

    def test_dephasing_rate_estimator(self):
        gamma_true = 0.2
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        coherence = [math.exp(-gamma_true * t) for t in times]
        gamma_est = estimate_logical_dephasing_rate(times, coherence)
        self.assertAlmostEqual(gamma_est, gamma_true, delta=0.03)
        g, lo, hi = estimate_logical_dephasing_rate_with_ci(times, coherence)
        self.assertAlmostEqual(g, gamma_true, delta=0.03)
        self.assertLessEqual(lo, g)
        self.assertLessEqual(g, hi)

    def test_effective_observation_error_and_emission(self):
        p0 = effective_observation_error(0.01, 0.01, 0.02)
        p1 = effective_observation_error(0.03, 0.01, 0.02)
        p2 = effective_observation_error(0.03, 0.03, 0.02)
        p3 = effective_observation_error(0.03, 0.03, 0.2)
        self.assertLess(p0, p1)
        self.assertLess(p1, p2)
        self.assertLess(p2, p3)

        # For state 2 (syndrome bits 1,1), matching observation should be likelier than mismatch.
        good = emission_probability((1, 1), 2, p_m=0.02, p_r=0.01, zeta=0.04)
        bad = emission_probability((0, 0), 2, p_m=0.02, p_r=0.01, zeta=0.04)
        self.assertGreater(good, bad)

    def test_bayes_filter_step_and_run_monitoring_filter(self):
        # Identity transition to isolate emission update.
        t_id = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        prior = [0.25, 0.25, 0.25, 0.25]
        post = bayes_filter_step(prior, t_id, (1, 1), p_m=0.01, p_r=0.0, zeta=0.0)
        self.assertAlmostEqual(sum(post), 1.0, places=10)
        self.assertEqual(len(post), 4)
        self.assertEqual(decode_map_correction_target(post), 1)  # syndrome -- => X2 target

        posts, states, targets = run_monitoring_filter(
            observations=[(1, 1), (1, 1), (1, 1)],
            transition=t_id,
            p_m=0.01,
            p_r=0.0,
            zeta=0.0,
            prior=[0.25, 0.25, 0.25, 0.25],
        )
        self.assertEqual(len(posts), 3)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[-1], 1)
        self.assertIn(1, targets)

    def test_decoder_transition_matrix_shape(self):
        p = EffectiveModelParams(gamma_x=0.02, chi=3.0, shots=40, max_rounds=30)
        t = build_decoder_transition_matrix(p, MODE_AUTO)
        self.assertEqual(len(t), 4)
        for row in t:
            self.assertEqual(len(row), 4)
            self.assertAlmostEqual(sum(row), 1.0, places=10)

    def test_sparse_measurement_schedule_helpers(self):
        self.assertFalse(is_sparse_measurement_round(0, 4))
        self.assertFalse(is_sparse_measurement_round(2, 4))
        self.assertTrue(is_sparse_measurement_round(3, 4))
        self.assertEqual(sparse_measurement_rounds(10, 4), (3, 7))
        self.assertEqual(sparse_measurement_count(10, 4), 2)
        self.assertAlmostEqual(total_time_for_rounds(10, 4, 1.2, 0.25), 12.6, places=10)

    def test_syndrome_sampling_from_rho(self):
        # |010> is an eigenstate with syndrome bits (1,1).
        ket = np.zeros((8,), dtype=complex)
        ket[2] = 1.0
        rho = np.outer(ket, ket.conj())
        rng = random.Random(55)

        ideal = [sample_ideal_syndrome_from_rho(rho, rng) for _ in range(100)]
        self.assertTrue(all(o == (1, 1) for o in ideal))

        noisy = [sample_noisy_syndrome_from_rho(rho, p_m=0.2, rng=rng) for _ in range(2500)]
        mismatch = sum(1 for o in noisy if o != (1, 1)) / float(len(noisy))
        self.assertGreater(mismatch, 0.28)
        self.assertLess(mismatch, 0.46)

    def test_monitoring_observation_pr_effect(self):
        rng = random.Random(99)
        n = 2400
        low = [sample_monitoring_observation_from_state(2, p_m=0.01, p_r=0.001, zeta=0.0, rng=rng) for _ in range(n)]
        high = [sample_monitoring_observation_from_state(2, p_m=0.01, p_r=0.1, zeta=0.0, rng=rng) for _ in range(n)]
        err_low = sum(1 for o in low if o != (1, 1)) / float(n)
        err_high = sum(1 for o in high if o != (1, 1)) / float(n)
        self.assertGreater(err_high, err_low + 0.04)

    def test_apply_reset_fault(self):
        rng = random.Random(202)
        # p_r=0: no corruption
        for _ in range(30):
            self.assertEqual(apply_reset_fault(0b000, p_r=0.0, rng=rng), 0b000)
        # p_r=1: always inject one X fault from 000 -> one-hot patterns
        got = {apply_reset_fault(0b000, p_r=1.0, rng=rng) for _ in range(80)}
        self.assertTrue(got.issubset({0b001, 0b010, 0b100}))
        self.assertTrue(len(got) >= 2)

    def test_lifetime_degrades_with_pr(self):
        base = EffectiveModelParams(
            gamma_x=0.03,
            chi=0.0,
            zeta=0.03,
            eta_t=0.1,
            p_m=0.01,
            k=6,
            shots=500,
            max_rounds=160,
            seed=17,
        )
        low = simulate_lifetime(replace(base, p_r=0.001), MODE_NAIVE)
        high = simulate_lifetime(replace(base, p_r=0.08), MODE_NAIVE)
        self.assertLess(high.lifetime_time, low.lifetime_time)


if __name__ == "__main__":
    unittest.main()
