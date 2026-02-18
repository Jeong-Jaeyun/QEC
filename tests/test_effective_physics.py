import unittest
from dataclasses import replace

import numpy as np

from core.effective_model import (
    MODE_FULL,
    MODE_NAIVE,
    EffectiveModelParams,
    build_round_superop,
    cptp_sanity_metrics,
    engineered_jump_operators,
    round_density_matrix_step,
    simulate_lifetime,
    syndrome_projector,
)


def _ket(index: int, dim: int = 8) -> np.ndarray:
    v = np.zeros((dim,), dtype=complex)
    v[index] = 1.0
    return v


def _random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    re = rng.normal(size=(dim,))
    im = rng.normal(size=(dim,))
    vec = re + 1j * im
    nrm = np.linalg.norm(vec)
    if nrm <= 0.0:
        vec[0] = 1.0
        nrm = 1.0
    return vec / nrm


def _hermiticity_residual(rho: np.ndarray) -> float:
    return float(np.linalg.norm(rho - rho.conj().T))


def _trace_residual(rho: np.ndarray) -> float:
    return abs(float(np.trace(rho).real) - 1.0)


def _min_eig_hermitized(rho: np.ndarray) -> float:
    rho_h = 0.5 * (rho + rho.conj().T)
    return float(np.min(np.linalg.eigvalsh(rho_h)).real)


class TestEffectivePhysics(unittest.TestCase):
    def test_syndrome_projectors(self):
        pis = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                p = syndrome_projector(s1, s2)
                pis.append(p)
                self.assertLess(np.linalg.norm(p.conj().T - p), 1e-10)
                self.assertLess(np.linalg.norm(p @ p - p), 1e-10)
        s = sum(pis)
        self.assertLess(np.linalg.norm(s - np.eye(8)), 1e-10)

    def test_engineered_jump_mappings(self):
        l1, l2, l3 = engineered_jump_operators(gamma=1.0)

        self.assertLess(np.linalg.norm(l1 @ _ket(4) - _ket(0)), 1e-10)
        self.assertLess(np.linalg.norm(l1 @ _ket(3) - _ket(7)), 1e-10)

        self.assertLess(np.linalg.norm(l2 @ _ket(2) - _ket(0)), 1e-10)
        self.assertLess(np.linalg.norm(l2 @ _ket(5) - _ket(7)), 1e-10)

        self.assertLess(np.linalg.norm(l3 @ _ket(1) - _ket(0)), 1e-10)
        self.assertLess(np.linalg.norm(l3 @ _ket(6) - _ket(7)), 1e-10)

    def test_round_map_cptp_sanity_and_state_validity(self):
        superop = build_round_superop(chi=2.0, gamma_x=0.02, tau_int=1.0)
        met = cptp_sanity_metrics(superop)
        self.assertLess(met["tp_error"], 1e-7)
        self.assertGreater(met["choi_min_eig"], -1e-7)

        ket_plus_l = (_ket(0) + _ket(7)) / np.sqrt(2.0)
        rho0 = np.outer(ket_plus_l, ket_plus_l.conj())
        rho1 = round_density_matrix_step(rho0, chi=2.0, gamma_x=0.02, tau_int=1.0, delta=0.03)
        self.assertAlmostEqual(float(np.real(np.trace(rho1))), 1.0, places=8)
        eigs = np.linalg.eigvalsh(0.5 * (rho1 + rho1.conj().T))
        self.assertGreater(float(np.min(np.real(eigs))), -1e-8)

    def test_round_invariants_per_step(self):
        # G1 required invariants:
        #   trace preservation, Hermiticity, relaxed positivity per round.
        rng = np.random.default_rng(1234)
        psi = _random_pure_state(8, rng)
        rho = np.outer(psi, psi.conj())

        for _ in range(40):
            delta = float(rng.normal(0.0, 0.03))
            rho = round_density_matrix_step(
                rho,
                chi=2.8,
                gamma_x=0.035,
                tau_int=1.0,
                delta=delta,
            )
            self.assertLess(_trace_residual(rho), 1e-9)
            self.assertLess(_hermiticity_residual(rho), 1e-9)
            self.assertGreaterEqual(_min_eig_hermitized(rho), -1e-10)

    def test_cptp_sanity_on_random_pure_states(self):
        # G1 simplified CPTP check:
        # map random pure states and ensure PSD + trace(1) + Hermiticity.
        rng = np.random.default_rng(2026)
        for _ in range(24):
            psi = _random_pure_state(8, rng)
            rho0 = np.outer(psi, psi.conj())
            delta = float(rng.normal(0.0, 0.02))
            rho1 = round_density_matrix_step(
                rho0,
                chi=3.0,
                gamma_x=0.02,
                tau_int=1.0,
                delta=delta,
            )
            self.assertLess(_trace_residual(rho1), 1e-9)
            self.assertLess(_hermiticity_residual(rho1), 1e-9)
            self.assertGreaterEqual(_min_eig_hermitized(rho1), -1e-10)

    def test_chi_zero_naive_degrades(self):
        p = EffectiveModelParams(
            gamma_x=0.03,
            chi=0.0,
            zeta=0.03,
            eta_t=0.05,
            p_m=0.001,
            p_r=0.001,
            k=6,
            shots=420,
            max_rounds=120,
            seed=101,
        )
        full = simulate_lifetime(replace(p, k=1), MODE_FULL)
        naive = simulate_lifetime(replace(p, k=6), MODE_NAIVE)
        self.assertLess(naive.lifetime_time, full.lifetime_time)

    def test_seed_reproducibility(self):
        p = EffectiveModelParams(shots=120, max_rounds=80, seed=44, k=5, chi=2.4, p_m=0.03, p_r=0.02)
        a = simulate_lifetime(p, MODE_NAIVE)
        b = simulate_lifetime(p, MODE_NAIVE)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
