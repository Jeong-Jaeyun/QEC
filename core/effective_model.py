from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Hidden-state labels used in the effective model:
#   0: ++  (no X error)
#   1: -+  (single X on data qubit 1)
#   2: --  (single X on data qubit 2)
#   3: +-  (single X on data qubit 3)
#   4: logical-failure manifold (double/triple X), treated as absorbing.

STATE_NO_ERROR = 0
STATE_X1 = 1
STATE_X2 = 2
STATE_X3 = 3
STATE_FAIL = 4

MODE_FULL = "full_syndrome"
MODE_NAIVE = "naive_sparse"
MODE_AUTO = "autonomous_sparse"
VALID_MODES = (MODE_FULL, MODE_NAIVE, MODE_AUTO)

STATE_TO_PATTERN = {
    STATE_NO_ERROR: 0b000,
    STATE_X1: 0b001,
    STATE_X2: 0b010,
    STATE_X3: 0b100,
}

PATTERN_TO_STATE = {
    0b000: STATE_NO_ERROR,
    0b001: STATE_X1,
    0b010: STATE_X2,
    0b100: STATE_X3,
}

# Syndrome mapping for S1=Z1Z2, S2=Z2Z3 where +1->0 and -1->1.
STATE_TO_SYNDROME = {
    STATE_NO_ERROR: (0, 0),  # ++
    STATE_X1: (1, 0),        # -+
    STATE_X2: (1, 1),        # --
    STATE_X3: (0, 1),        # +-
}


@dataclass(frozen=True)
class EffectiveModelParams:
    """
    Effective dynamics parameters aligned with QEC_develop.md final spec.

    Units:
    - gamma_x: physical X error rate (1 / time)
    - tau_int: autonomous interaction window duration
    - chi    : dimensionless autonomous strength = Gamma_eff * tau_int
    - zeta   : dimensionless drift strength       = sigma_z * tau_int
    - eta_t  : dimensionless measurement overhead = (t_m + t_r) / tau_int
    """

    gamma_x: float = 0.03
    chi: float = 4.0
    zeta: float = 0.05
    sigma_z: Optional[float] = None
    eta_t: float = 0.3
    t_m: Optional[float] = None
    t_r: Optional[float] = None
    tau_int: float = 1.0

    p_m: float = 0.01
    p_r: float = 0.01

    k: int = 4
    max_rounds: int = 300
    shots: int = 1200
    seed: int = 11


@dataclass(frozen=True)
class LifetimeStats:
    mode: str
    k: int
    shots: int
    max_rounds: int
    logical_failure_prob: float
    survival_prob: float
    lifetime_time: float
    lifetime_time_median: float
    lifetime_time_ci_low: float
    lifetime_time_ci_high: float
    lifetime_rounds: float
    avg_measurements_to_fail: float
    measurement_efficiency: float


@dataclass(frozen=True)
class KSweepPoint:
    k: int
    lifetime_time: float
    measurement_efficiency: float
    gain_vs_full: float
    measurement_gain_vs_full: float
    logical_failure_prob: float


@dataclass(frozen=True)
class KSweepResult:
    mode: str
    k_star: int
    best_stats: LifetimeStats
    best_gain: float
    best_measurement_gain: float
    points: Tuple[KSweepPoint, ...]


@dataclass(frozen=True)
class CoherencePoint:
    round_index: int
    time: float
    coherence: float


@dataclass(frozen=True)
class CoherenceResult:
    mode: str
    k: int
    gamma_lphi: float
    gamma_lphi_ci_low: float
    gamma_lphi_ci_high: float
    points: Tuple[CoherencePoint, ...]


@dataclass(frozen=True)
class BaselineBundle:
    full: LifetimeStats
    naive: LifetimeStats
    autonomous: LifetimeStats


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_params(p: EffectiveModelParams) -> EffectiveModelParams:
    zeta = float(p.zeta)
    if p.sigma_z is not None:
        zeta = float(p.sigma_z) * float(p.tau_int)

    eta_t = float(p.eta_t)
    if p.t_m is not None or p.t_r is not None:
        t_m = float(p.t_m if p.t_m is not None else 0.0)
        t_r = float(p.t_r if p.t_r is not None else 0.0)
        eta_t = (t_m + t_r) / float(p.tau_int)

    sigma_z = float(p.sigma_z) if p.sigma_z is not None else (zeta / float(p.tau_int))
    t_m = float(p.t_m) if p.t_m is not None else 0.0
    t_r = float(p.t_r) if p.t_r is not None else 0.0
    return replace(p, zeta=zeta, sigma_z=sigma_z, eta_t=eta_t, t_m=t_m, t_r=t_r)


def _assert_params(p: EffectiveModelParams) -> None:
    if p.sigma_z is not None and p.sigma_z < 0.0:
        raise ValueError("sigma_z must be >= 0")
    if p.gamma_x < 0.0:
        raise ValueError("gamma_x must be >= 0")
    if p.chi < 0.0:
        raise ValueError("chi must be >= 0")
    if p.zeta < 0.0:
        raise ValueError("zeta must be >= 0")
    if p.eta_t < 0.0:
        raise ValueError("eta_t must be >= 0")
    if p.tau_int <= 0.0:
        raise ValueError("tau_int must be > 0")
    if p.t_m is not None and p.t_m < 0.0:
        raise ValueError("t_m must be >= 0")
    if p.t_r is not None and p.t_r < 0.0:
        raise ValueError("t_r must be >= 0")
    if not (0.0 <= p.p_m <= 0.5):
        raise ValueError("p_m must be in [0, 0.5]")
    if not (0.0 <= p.p_r <= 0.5):
        raise ValueError("p_r must be in [0, 0.5]")
    if p.k <= 0:
        raise ValueError("k must be >= 1")
    if p.max_rounds <= 0:
        raise ValueError("max_rounds must be >= 1")
    if p.shots <= 0:
        raise ValueError("shots must be >= 1")


def syndrome_for_state(state: int) -> Tuple[int, int]:
    if state not in STATE_TO_SYNDROME:
        raise ValueError(f"state must be 0..3, got {state}")
    return STATE_TO_SYNDROME[state]


def _pattern_to_state(pattern: int) -> int:
    return PATTERN_TO_STATE.get(pattern, STATE_FAIL)


def _state_to_pattern(state: int) -> int:
    if state not in STATE_TO_PATTERN:
        raise ValueError(f"state must be 0..3, got {state}")
    return STATE_TO_PATTERN[state]


def _hamming_weight3(x: int) -> int:
    return (x & 1) + ((x >> 1) & 1) + ((x >> 2) & 1)


def _effective_k(mode: str, k: int) -> int:
    if mode == MODE_FULL:
        return 1
    return max(1, int(k))


def is_sparse_measurement_round(round_index: int, k: int) -> bool:
    """
    Return True when sparse monitoring should run at this 0-based round index.
    Convention: first measurement happens after k interactions, i.e. (r+1) % k == 0.
    """
    if int(round_index) < 0:
        raise ValueError("round_index must be >= 0")
    if int(k) <= 0:
        raise ValueError("k must be >= 1")
    return ((int(round_index) + 1) % int(k)) == 0


def sparse_measurement_rounds(n_rounds: int, k: int) -> Tuple[int, ...]:
    if int(n_rounds) < 0:
        raise ValueError("n_rounds must be >= 0")
    return tuple(r for r in range(int(n_rounds)) if is_sparse_measurement_round(r, k))


def sparse_measurement_count(rounds: int, k: int) -> int:
    if int(rounds) < 0:
        raise ValueError("rounds must be >= 0")
    if int(k) <= 0:
        raise ValueError("k must be >= 1")
    return int(rounds) // int(k)


def _bit_flip_probability(gamma_x: float, tau_int: float) -> float:
    # Bernoulli parameter after one interaction window.
    return 1.0 - math.exp(-float(gamma_x) * float(tau_int))


def _autonomous_correction_probability(chi: float, mode: str) -> float:
    if mode != MODE_AUTO:
        return 0.0
    return 1.0 - math.exp(-float(chi))


def _syndrome_bit_error_prob(p: EffectiveModelParams) -> float:
    return effective_observation_error(float(p.p_m), float(p.p_r), float(p.zeta))


def effective_observation_error(p_m: float, p_r: float, zeta: float) -> float:
    """
    Effective per-syndrome-bit observation error used by sparse monitoring.
    This folds readout (`p_m`), reset contamination (`p_r`), and drift blur (`zeta`).
    """
    p_err = float(p_m) + 0.35 * float(p_r) + 0.22 * min(1.0, float(zeta))
    return _clamp(p_err, 0.0, 0.49)


def _measurement_backaction_factor(p: EffectiveModelParams) -> float:
    # Coherence penalty per measurement event.
    fac = 1.0 - 1.6 * float(p.p_m) - 0.9 * float(p.p_r)
    return _clamp(fac, 0.0, 1.0)


def _autonomous_dephasing_rate(p: EffectiveModelParams, mode: str) -> float:
    if mode != MODE_AUTO:
        return 0.0
    # Weak penalty: stronger autonomous pumping can add phase disturbance.
    chi = float(p.chi)
    return 0.002 * (chi * chi) / (1.0 + 0.5 * chi)


def _round_time_increment(p: EffectiveModelParams, mode: str, round_index: int) -> float:
    dt = float(p.tau_int)
    k_eff = _effective_k(mode, p.k)
    if is_sparse_measurement_round(round_index, k_eff):
        dt += float(p.eta_t) * float(p.tau_int)
    return dt


def total_time_for_rounds(rounds: int, k: int, tau_int: float, eta_t: float) -> float:
    if rounds < 0:
        raise ValueError("rounds must be >= 0")
    if k <= 0:
        raise ValueError("k must be >= 1")
    measurements = sparse_measurement_count(rounds, k)
    return float(rounds) * float(tau_int) + float(measurements) * float(eta_t) * float(tau_int)


def _id2() -> np.ndarray:
    return np.eye(2, dtype=complex)


def _x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def pauli_x_on(qubit: int) -> np.ndarray:
    if qubit == 1:
        return _kron3(_x(), _id2(), _id2())
    if qubit == 2:
        return _kron3(_id2(), _x(), _id2())
    if qubit == 3:
        return _kron3(_id2(), _id2(), _x())
    raise ValueError("qubit must be one of 1,2,3")


def pauli_z_on(qubit: int) -> np.ndarray:
    if qubit == 1:
        return _kron3(_z(), _id2(), _id2())
    if qubit == 2:
        return _kron3(_id2(), _z(), _id2())
    if qubit == 3:
        return _kron3(_id2(), _id2(), _z())
    raise ValueError("qubit must be one of 1,2,3")


def stabilizer_operators() -> Tuple[np.ndarray, np.ndarray]:
    s1 = _kron3(_z(), _z(), _id2())
    s2 = _kron3(_id2(), _z(), _z())
    return s1, s2


def syndrome_projector(s1_sign: int, s2_sign: int) -> np.ndarray:
    if s1_sign not in (-1, 1) or s2_sign not in (-1, 1):
        raise ValueError("s1_sign and s2_sign must be +1 or -1")
    s1, s2 = stabilizer_operators()
    eye = np.eye(8, dtype=complex)
    return 0.25 * (eye + float(s1_sign) * s1) @ (eye + float(s2_sign) * s2)


def projector_for_hidden_state(state: int) -> np.ndarray:
    if state == STATE_NO_ERROR:
        return syndrome_projector(+1, +1)
    if state == STATE_X1:
        return syndrome_projector(-1, +1)
    if state == STATE_X2:
        return syndrome_projector(-1, -1)
    if state == STATE_X3:
        return syndrome_projector(+1, -1)
    raise ValueError("state must be one of 0,1,2,3")


def engineered_jump_operators(gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")
    pref = math.sqrt(float(gamma))
    l1 = pref * pauli_x_on(1) @ syndrome_projector(-1, +1)
    l2 = pref * pauli_x_on(2) @ syndrome_projector(-1, -1)
    l3 = pref * pauli_x_on(3) @ syndrome_projector(+1, -1)
    return l1, l2, l3


def vec_density(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec_density(v: np.ndarray, dim: int) -> np.ndarray:
    return np.asarray(v, dtype=complex).reshape((dim, dim), order="F")


def lindblad_dissipator_superop(l_op: np.ndarray) -> np.ndarray:
    d = l_op.shape[0]
    eye = np.eye(d, dtype=complex)
    ld_l = l_op.conj().T @ l_op
    term_jump = np.kron(l_op.conj(), l_op)
    term_left = np.kron(eye, ld_l)
    term_right = np.kron(ld_l.T, eye)
    return term_jump - 0.5 * (term_left + term_right)


def liouvillian_correction(chi: float, tau_int: float) -> np.ndarray:
    if tau_int <= 0.0:
        raise ValueError("tau_int must be > 0")
    gamma = float(chi) / float(tau_int)
    l1, l2, l3 = engineered_jump_operators(gamma)
    return lindblad_dissipator_superop(l1) + lindblad_dissipator_superop(l2) + lindblad_dissipator_superop(l3)


def liouvillian_bitflip(gamma_x: float) -> np.ndarray:
    if gamma_x < 0.0:
        raise ValueError("gamma_x must be >= 0")
    return float(gamma_x) * (
        lindblad_dissipator_superop(pauli_x_on(1))
        + lindblad_dissipator_superop(pauli_x_on(2))
        + lindblad_dissipator_superop(pauli_x_on(3))
    )


def superop_expm(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(m)
    inv = np.linalg.inv(vecs)
    return vecs @ np.diag(np.exp(vals)) @ inv


def z_drift_unitary(delta: float, tau_int: float) -> np.ndarray:
    zsum = pauli_z_on(1) + pauli_z_on(2) + pauli_z_on(3)
    h = 0.5 * float(delta) * zsum
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1j * float(tau_int) * vals)) @ vecs.conj().T


def sample_ideal_syndrome_probs(rho: np.ndarray) -> Dict[Tuple[int, int], float]:
    probs: Dict[Tuple[int, int], float] = {}
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pi = syndrome_projector(s1, s2)
            p = float(np.real(np.trace(pi @ rho)))
            probs[(s1, s2)] = max(0.0, p)
    z = sum(probs.values())
    if z <= 0.0:
        return {(-1, -1): 0.25, (-1, 1): 0.25, (1, -1): 0.25, (1, 1): 0.25}
    return {k: v / z for k, v in probs.items()}


def sample_ideal_syndrome_from_rho(rho: np.ndarray, rng: random.Random) -> Tuple[int, int]:
    """
    Sample ideal stabilizer syndrome bits from rho without measurement bit flips.
    Returns bits in {0,1}x{0,1} for (S1,S2) where +1->0 and -1->1.
    """
    probs = sample_ideal_syndrome_probs(rho)
    r = rng.random()
    c = 0.0
    selected = (1, 1)
    for key in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        c += probs[key]
        if r <= c:
            selected = key
            break
    b0 = 0 if selected[0] == 1 else 1
    b1 = 0 if selected[1] == 1 else 1
    return b0, b1


def sample_noisy_syndrome_from_rho(rho: np.ndarray, p_m: float, rng: random.Random) -> Tuple[int, int]:
    b0, b1 = sample_ideal_syndrome_from_rho(rho, rng)
    pm = _clamp(float(p_m), 0.0, 0.5)
    if rng.random() < pm:
        b0 ^= 1
    if rng.random() < pm:
        b1 ^= 1
    return b0, b1


def apply_superop_to_rho(superop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    d = rho.shape[0]
    v = vec_density(rho)
    out = superop @ v
    return unvec_density(out, d)


def build_round_superop(chi: float, gamma_x: float, tau_int: float) -> np.ndarray:
    l = liouvillian_correction(chi, tau_int) + liouvillian_bitflip(gamma_x)
    return superop_expm(float(tau_int) * l)


def round_density_matrix_step(
    rho: np.ndarray,
    *,
    chi: float,
    gamma_x: float,
    tau_int: float,
    delta: float,
) -> np.ndarray:
    evo = build_round_superop(chi=chi, gamma_x=gamma_x, tau_int=tau_int)
    out = apply_superop_to_rho(evo, rho)
    uz = z_drift_unitary(delta=delta, tau_int=tau_int)
    return uz @ out @ uz.conj().T


def _basis_op(dim: int, i: int, j: int) -> np.ndarray:
    out = np.zeros((dim, dim), dtype=complex)
    out[i, j] = 1.0
    return out


def superop_to_choi(superop: np.ndarray) -> np.ndarray:
    d2 = int(superop.shape[0])
    d = int(round(math.sqrt(d2)))
    if d * d != d2:
        raise ValueError("Superoperator dimension must be square of Hilbert dimension.")
    j = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for k in range(d):
            eik = _basis_op(d, i, k)
            out = apply_superop_to_rho(superop, eik)
            j += np.kron(out, eik)
    return j


def cptp_sanity_metrics(superop: np.ndarray) -> Dict[str, float]:
    d2 = int(superop.shape[0])
    d = int(round(math.sqrt(d2)))
    eye = np.eye(d, dtype=complex)
    vec_i = vec_density(eye)
    tp_err = float(np.linalg.norm(vec_i.conj().T @ superop - vec_i.conj().T))

    choi = superop_to_choi(superop)
    choi_h = 0.5 * (choi + choi.conj().T)
    evals = np.linalg.eigvalsh(choi_h)
    min_eval = float(np.min(np.real(evals)))
    return {"tp_error": tp_err, "choi_min_eig": min_eval}


def build_full_transition_matrix(p: EffectiveModelParams, mode: str) -> List[List[float]]:
    """
    5x5 transition matrix including absorbing failure manifold.
    State order: [0,1,2,3,FAIL].
    """
    p = _normalize_params(p)
    _assert_params(p)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode={mode!r}. Use one of {VALID_MODES}.")

    p_x = _bit_flip_probability(p.gamma_x, p.tau_int)
    p_corr = _autonomous_correction_probability(p.chi, mode)

    mat = [[0.0 for _ in range(5)] for _ in range(5)]

    for s in range(4):
        pattern = _state_to_pattern(s)
        # Natural noise.
        natural_row = [0.0 for _ in range(5)]
        for mask in range(8):
            w = _hamming_weight3(mask)
            prob = (p_x ** w) * ((1.0 - p_x) ** (3 - w))
            nxt_pattern = pattern ^ mask
            nxt_state = _pattern_to_state(nxt_pattern)
            natural_row[nxt_state] += prob

        # Autonomous correction (single-error -> no-error).
        for mid_state, mid_prob in enumerate(natural_row):
            if mid_prob <= 0.0:
                continue
            if mid_state in (STATE_X1, STATE_X2, STATE_X3):
                mat[s][STATE_NO_ERROR] += mid_prob * p_corr
                mat[s][mid_state] += mid_prob * (1.0 - p_corr)
            else:
                mat[s][mid_state] += mid_prob

    mat[STATE_FAIL][STATE_FAIL] = 1.0
    return mat


def build_decoder_transition_matrix(p: EffectiveModelParams, mode: str) -> List[List[float]]:
    """
    4x4 transition matrix used by HMM/Bayes filter over hidden cosets {++, -+, --, +-}.
    Failure mass is dropped then rows are renormalized.
    """
    p = _normalize_params(p)
    full = build_full_transition_matrix(p, mode)
    out = [[0.0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        row_mass = sum(full[i][j] for j in range(4))
        if row_mass <= 0.0:
            out[i][STATE_NO_ERROR] = 1.0
            continue
        for j in range(4):
            out[i][j] = full[i][j] / row_mass
    return out


def _matvec_4(v: Sequence[float], m: Sequence[Sequence[float]]) -> List[float]:
    out = [0.0, 0.0, 0.0, 0.0]
    for j in range(4):
        s = 0.0
        for i in range(4):
            s += float(v[i]) * float(m[i][j])
        out[j] = s
    return out


def _sample_physical_noise(pattern: int, p_x: float, rng: random.Random) -> int:
    mask = 0
    if rng.random() < p_x:
        mask ^= 0b001
    if rng.random() < p_x:
        mask ^= 0b010
    if rng.random() < p_x:
        mask ^= 0b100
    return pattern ^ mask


def _sample_observation_bits(true_bits: Tuple[int, int], p_err: float, rng: random.Random) -> Tuple[int, int]:
    b0, b1 = true_bits
    if rng.random() < p_err:
        b0 ^= 1
    if rng.random() < p_err:
        b1 ^= 1
    return (b0, b1)


def sample_monitoring_observation_from_state(
    hidden_state: int,
    *,
    p_m: float,
    p_r: float,
    zeta: float,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Sample sparse-monitoring observation y from hidden state h.
    This is the C1/C2 observation model entry-point.
    """
    true_bits = syndrome_for_state(hidden_state)
    p_err = effective_observation_error(p_m, p_r, zeta)
    return _sample_observation_bits(true_bits, p_err, rng)


def sample_monitoring_observation_from_rho(
    rho: np.ndarray,
    *,
    p_m: float,
    p_r: float,
    zeta: float,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Sample sparse-monitoring observation y from rho.
    Ideal syndrome is sampled from projector probabilities, then observation
    noise is applied via effective_observation_error(p_m,p_r,zeta).
    """
    true_bits = sample_ideal_syndrome_from_rho(rho, rng)
    p_err = effective_observation_error(p_m, p_r, zeta)
    return _sample_observation_bits(true_bits, p_err, rng)


def _emission_probability(obs: Tuple[int, int], hidden_state: int, p_err: float) -> float:
    t0, t1 = syndrome_for_state(hidden_state)
    o0, o1 = obs
    p0 = (1.0 - p_err) if o0 == t0 else p_err
    p1 = (1.0 - p_err) if o1 == t1 else p_err
    return p0 * p1


def emission_probability(
    obs: Tuple[int, int],
    hidden_state: int,
    *,
    p_m: float,
    p_r: float,
    zeta: float,
) -> float:
    """
    Public emission model O(y|h) for monitoring decoder diagnostics.
    """
    p_err = effective_observation_error(p_m, p_r, zeta)
    return _emission_probability(obs, hidden_state, p_err)


def _normalize_prob(v: Sequence[float]) -> List[float]:
    z = sum(v)
    if z <= 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    return [float(x) / z for x in v]


def bayes_filter_step(
    prior: Sequence[float],
    transition: Sequence[Sequence[float]],
    observation: Tuple[int, int],
    *,
    p_m: float,
    p_r: float,
    zeta: float,
    predict: bool = True,
) -> List[float]:
    """
    Single online Bayes-filter update:
      posterior ~ O(y|h) * sum_h' T(h'->h) * prior(h')
    """
    pred = _matvec_4(prior, transition) if predict else list(prior)
    p_err = effective_observation_error(p_m, p_r, zeta)
    unnorm = [pred[i] * _emission_probability(observation, i, p_err) for i in range(4)]
    return _normalize_prob(unnorm)


def decode_map_state(posterior: Sequence[float]) -> int:
    if len(posterior) != 4:
        raise ValueError("posterior must have length 4")
    return int(max(range(4), key=lambda idx: float(posterior[idx])))


def decode_map_correction_target(posterior: Sequence[float]) -> Optional[int]:
    return _map_state_to_correction_target(decode_map_state(posterior))


def run_monitoring_filter(
    observations: Sequence[Tuple[int, int]],
    transition: Sequence[Sequence[float]],
    *,
    p_m: float,
    p_r: float,
    zeta: float,
    prior: Optional[Sequence[float]] = None,
) -> Tuple[Tuple[List[float], ...], Tuple[int, ...], Tuple[Optional[int], ...]]:
    """
    Run online Bayes filtering over a sequence of syndrome observations.
    Returns:
      - posterior sequence
      - MAP hidden-state sequence
      - MAP correction-target sequence
    """
    cur = list(prior) if prior is not None else [1.0, 0.0, 0.0, 0.0]
    cur = _normalize_prob(cur)

    posteriors: List[List[float]] = []
    map_states: List[int] = []
    map_targets: List[Optional[int]] = []
    for obs in observations:
        cur = bayes_filter_step(cur, transition, obs, p_m=p_m, p_r=p_r, zeta=zeta)
        map_state = decode_map_state(cur)
        posteriors.append(list(cur))
        map_states.append(map_state)
        map_targets.append(_map_state_to_correction_target(map_state))
    return tuple(posteriors), tuple(map_states), tuple(map_targets)


def _map_state_to_correction_target(state: int) -> Optional[int]:
    if state == STATE_X1:
        return 0
    if state == STATE_X2:
        return 1
    if state == STATE_X3:
        return 2
    return None


def _apply_correction(pattern: int, target: Optional[int]) -> int:
    if target is None:
        return pattern
    if target == 0:
        return pattern ^ 0b001
    if target == 1:
        return pattern ^ 0b010
    if target == 2:
        return pattern ^ 0b100
    raise ValueError(f"target must be None or 0/1/2, got {target!r}")


def _inject_reset_fault(pattern: int, p_r: float, rng: random.Random) -> int:
    if rng.random() >= p_r:
        return pattern
    target = rng.randrange(3)
    return _apply_correction(pattern, target)


def apply_reset_fault(pattern: int, p_r: float, rng: random.Random) -> int:
    """
    Public wrapper for reset-fault injection used in sparse-monitoring updates.
    """
    return _inject_reset_fault(pattern, p_r, rng)


@dataclass(frozen=True)
class _TrajectoryResult:
    failed: bool
    fail_round: int
    fail_time: float
    n_measurements: int


def _mean_and_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    n = len(values)
    if n <= 0:
        return 0.0, 0.0, 0.0
    mean = float(sum(values) / float(n))
    if n == 1:
        return mean, mean, mean
    var = float(sum((float(x) - mean) ** 2 for x in values) / float(n - 1))
    sem = math.sqrt(max(var, 0.0) / float(n))
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _run_single_trajectory(
    p: EffectiveModelParams,
    mode: str,
    rng: random.Random,
) -> _TrajectoryResult:
    p = _normalize_params(p)
    k_eff = _effective_k(mode, p.k)
    p_x = _bit_flip_probability(p.gamma_x, p.tau_int)
    p_corr = _autonomous_correction_probability(p.chi, mode)

    decoder_t = build_decoder_transition_matrix(p, mode)
    posterior = [1.0, 0.0, 0.0, 0.0]

    pattern = 0b000
    elapsed = 0.0
    n_meas = 0

    for r in range(1, int(p.max_rounds) + 1):
        pattern = _sample_physical_noise(pattern, p_x, rng)
        state = _pattern_to_state(pattern)

        if state in (STATE_X1, STATE_X2, STATE_X3) and p_corr > 0.0 and rng.random() < p_corr:
            pattern = 0b000
            state = STATE_NO_ERROR

        elapsed += float(p.tau_int)
        posterior = _matvec_4(posterior, decoder_t)
        posterior = _normalize_prob(posterior)

        if state == STATE_FAIL:
            return _TrajectoryResult(True, r, elapsed, n_meas)

        if is_sparse_measurement_round(r - 1, k_eff):
            n_meas += 1
            elapsed += float(p.eta_t) * float(p.tau_int)

            obs = sample_monitoring_observation_from_state(
                state,
                p_m=p.p_m,
                p_r=p.p_r,
                zeta=p.zeta,
                rng=rng,
            )
            posterior = bayes_filter_step(
                posterior,
                decoder_t,
                obs,
                p_m=p.p_m,
                p_r=p.p_r,
                zeta=p.zeta,
                predict=False,
            )

            target = decode_map_correction_target(posterior)
            pattern = _apply_correction(pattern, target)
            pattern = _inject_reset_fault(pattern, p.p_r, rng)
            state = _pattern_to_state(pattern)

            # Decoder updates are sparse; after an explicit correction pulse we restart
            # from "best-known" code-space prior.
            posterior = [1.0, 0.0, 0.0, 0.0]

            if state == STATE_FAIL:
                return _TrajectoryResult(True, r, elapsed, n_meas)

    return _TrajectoryResult(False, int(p.max_rounds), elapsed, n_meas)


def simulate_lifetime(p: EffectiveModelParams, mode: str) -> LifetimeStats:
    p = _normalize_params(p)
    _assert_params(p)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode={mode!r}. Use one of {VALID_MODES}.")

    k_eff = _effective_k(mode, p.k)
    fail_times: List[float] = []
    fail_rounds: List[float] = []
    meas_counts: List[float] = []
    n_fail = 0

    for shot in range(int(p.shots)):
        rng = random.Random(int(p.seed) + 7919 * shot + 101 * k_eff + len(mode))
        tr = _run_single_trajectory(p, mode, rng)
        if tr.failed:
            n_fail += 1
        fail_times.append(float(tr.fail_time))
        fail_rounds.append(float(tr.fail_round))
        meas_counts.append(float(tr.n_measurements))

    shots = float(p.shots)
    logical_failure_prob = n_fail / shots
    survival_prob = 1.0 - logical_failure_prob
    lifetime_time, lifetime_time_ci_low, lifetime_time_ci_high = _mean_and_ci95(fail_times)
    lifetime_time_median = _median(fail_times)
    lifetime_rounds = sum(fail_rounds) / shots
    avg_measurements = sum(meas_counts) / shots
    measurement_efficiency = lifetime_time / max(avg_measurements, 1e-9)

    return LifetimeStats(
        mode=mode,
        k=k_eff,
        shots=int(p.shots),
        max_rounds=int(p.max_rounds),
        logical_failure_prob=logical_failure_prob,
        survival_prob=survival_prob,
        lifetime_time=lifetime_time,
        lifetime_time_median=lifetime_time_median,
        lifetime_time_ci_low=lifetime_time_ci_low,
        lifetime_time_ci_high=lifetime_time_ci_high,
        lifetime_rounds=lifetime_rounds,
        avg_measurements_to_fail=avg_measurements,
        measurement_efficiency=measurement_efficiency,
    )


def run_three_baselines(p: EffectiveModelParams, *, k_sparse: Optional[int] = None) -> BaselineBundle:
    p = _normalize_params(p)
    _assert_params(p)
    k_use = int(k_sparse if k_sparse is not None else p.k)
    if k_use <= 1:
        raise ValueError("k_sparse must be > 1 for naive/autonomous sparse baselines.")

    full = simulate_lifetime(replace(p, k=1), MODE_FULL)
    naive = simulate_lifetime(replace(p, k=k_use), MODE_NAIVE)
    autonomous = simulate_lifetime(replace(p, k=k_use), MODE_AUTO)
    return BaselineBundle(full=full, naive=naive, autonomous=autonomous)


def sweep_k(
    p: EffectiveModelParams,
    mode: str,
    k_values: Iterable[int],
    *,
    full_stats: Optional[LifetimeStats] = None,
) -> KSweepResult:
    p = _normalize_params(p)
    if mode not in (MODE_NAIVE, MODE_AUTO):
        raise ValueError("sweep_k mode must be naive_sparse or autonomous_sparse.")

    k_list = sorted(set(int(k) for k in k_values if int(k) >= 1))
    if not k_list:
        raise ValueError("k_values must contain at least one k >= 1.")

    if full_stats is None:
        full_stats = simulate_lifetime(replace(p, k=1), MODE_FULL)

    points: List[KSweepPoint] = []
    best_key: Optional[Tuple[float, float]] = None
    best_stats: Optional[LifetimeStats] = None
    best_gain = 0.0
    best_mgain = 0.0
    best_k = k_list[0]

    for k in k_list:
        stats = simulate_lifetime(replace(p, k=k), mode)
        gain = stats.lifetime_time / max(full_stats.lifetime_time, 1e-12)
        mgain = stats.measurement_efficiency / max(full_stats.measurement_efficiency, 1e-12)
        points.append(
            KSweepPoint(
                k=stats.k,
                lifetime_time=stats.lifetime_time,
                measurement_efficiency=stats.measurement_efficiency,
                gain_vs_full=gain,
                measurement_gain_vs_full=mgain,
                logical_failure_prob=stats.logical_failure_prob,
            )
        )
        # Primary objective: maximize lifetime; tie-breaker: smaller k.
        key = (stats.lifetime_time, -float(stats.k))
        if best_key is None or key > best_key:
            best_key = key
            best_stats = stats
            best_gain = gain
            best_mgain = mgain
            best_k = stats.k

    assert best_stats is not None
    return KSweepResult(
        mode=mode,
        k_star=best_k,
        best_stats=best_stats,
        best_gain=best_gain,
        best_measurement_gain=best_mgain,
        points=tuple(points),
    )


def simulate_logical_coherence(
    p: EffectiveModelParams,
    mode: str,
    sample_rounds: Sequence[int],
    *,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
) -> CoherenceResult:
    """
    Coherence track for |+_L> using C_L(t)=<X_L>, X_L=X1X2X3.
    Drift model: quasi-static Gaussian with sigma_z = zeta / tau_int.
    """
    p = _normalize_params(p)
    _assert_params(p)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode={mode!r}. Use one of {VALID_MODES}.")

    rounds = sorted(set(int(r) for r in sample_rounds if int(r) > 0))
    if not rounds:
        raise ValueError("sample_rounds must contain at least one positive integer.")

    k_eff = _effective_k(mode, p.k)
    max_round = max(rounds)
    round_set = set(rounds)

    n_shots = int(shots if shots is not None else p.shots)
    base_seed = int(seed if seed is not None else p.seed)
    sigma_z = float(p.zeta) / float(p.tau_int)
    auto_gamma = _autonomous_dephasing_rate(p, mode)
    meas_fac = _measurement_backaction_factor(p)

    accum = {r: 0.0 for r in rounds}
    for shot in range(n_shots):
        rng = random.Random(base_seed + 104729 * shot + 17 * k_eff + len(mode))
        delta = rng.gauss(0.0, sigma_z)
        amp = complex(1.0, 0.0)

        for r in range(1, max_round + 1):
            dt_int = float(p.tau_int)
            phase_int = 3.0 * delta * dt_int
            amp *= complex(math.cos(phase_int), math.sin(phase_int))
            amp *= math.exp(-auto_gamma * dt_int)

            if is_sparse_measurement_round(r - 1, k_eff):
                dt_meas = float(p.eta_t) * float(p.tau_int)
                if dt_meas > 0.0:
                    phase_meas = 3.0 * delta * dt_meas
                    amp *= complex(math.cos(phase_meas), math.sin(phase_meas))
                    amp *= math.exp(-auto_gamma * dt_meas)
                amp *= meas_fac

            if r in round_set:
                accum[r] += float(amp.real)

    points: List[CoherencePoint] = []
    for r in rounds:
        c = accum[r] / float(n_shots)
        t = total_time_for_rounds(r, k_eff, p.tau_int, p.eta_t)
        points.append(CoherencePoint(round_index=r, time=t, coherence=c))

    gamma, gamma_ci_low, gamma_ci_high = estimate_logical_dephasing_rate_with_ci(
        [pt.time for pt in points],
        [pt.coherence for pt in points],
    )
    return CoherenceResult(
        mode=mode,
        k=k_eff,
        gamma_lphi=gamma,
        gamma_lphi_ci_low=gamma_ci_low,
        gamma_lphi_ci_high=gamma_ci_high,
        points=tuple(points),
    )


def estimate_logical_dephasing_rate(times: Sequence[float], coherence: Sequence[float]) -> float:
    gamma, _, _ = estimate_logical_dephasing_rate_with_ci(times, coherence)
    return gamma


def estimate_logical_dephasing_rate_with_ci(
    times: Sequence[float],
    coherence: Sequence[float],
) -> Tuple[float, float, float]:
    if len(times) != len(coherence):
        raise ValueError("times and coherence must have the same length.")
    if len(times) < 2:
        return 0.0, 0.0, 0.0

    xs: List[float] = []
    ys: List[float] = []
    for t, c in zip(times, coherence):
        ct = abs(float(c))
        if ct <= 1e-12:
            continue
        xs.append(float(t))
        ys.append(math.log(ct))

    if len(xs) < 2:
        return 0.0, 0.0, 0.0

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0.0:
        return 0.0, 0.0, 0.0
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    gamma = max(0.0, -slope)

    if len(xs) <= 2:
        return gamma, gamma, gamma
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, ys)]
    rss = sum(r * r for r in residuals)
    sigma2 = rss / max(1.0, float(len(xs) - 2))
    se_slope = math.sqrt(max(0.0, sigma2) / var_x)
    ci_delta = 1.96 * se_slope
    slope_low = slope - ci_delta
    slope_high = slope + ci_delta
    gamma_low = max(0.0, -slope_high)
    gamma_high = max(0.0, -slope_low)
    return gamma, gamma_low, gamma_high


def is_safe_zone(
    lifetime_ratio: float,
    gamma_ratio: float,
    *,
    epsilon_tau: float = 0.1,
    epsilon_phi: float = 0.1,
) -> bool:
    return (float(lifetime_ratio) > 1.0 + float(epsilon_tau)) and (float(gamma_ratio) < 1.0 + float(epsilon_phi))


def measurement_normalized_gain(stats: LifetimeStats, full_stats: LifetimeStats) -> float:
    return float(stats.measurement_efficiency) / max(float(full_stats.measurement_efficiency), 1e-12)


def lifetime_gain(stats: LifetimeStats, full_stats: LifetimeStats) -> float:
    return float(stats.lifetime_time) / max(float(full_stats.lifetime_time), 1e-12)
