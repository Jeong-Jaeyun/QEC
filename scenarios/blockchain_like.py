from __future__ import annotations

"""
Scenario 3: "blockchain-like" consensus decoders for repetition-code syndrome streams.

This file provides two different approaches:

1) Classical consensus (measured syndrome per cycle):
   - Use core.circuit.build_core_circuit_with_syndrome() to log syndrome bits.
   - Decode using flat majority or block-consensus on the stabilizer streams.

2) Quantum memory consensus (no per-cycle syndrome measurement):
   - Treat the buffer qubit as a small quantum memory that is updated every cycle
     by parity-dependent phase kicks and a fixed rotation.
   - The final buffer measurement is then used as a "consensus bit" in post-processing.

Note: qiskit-aer is OPTIONAL here. Without Aer, we can still run ideal (no-noise) statevector sampling
via BasicSimulator.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env_check import require_typing_extensions_self

require_typing_extensions_self(package="qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers.basic_provider import BasicSimulator

from core.circuit import CoreParams, encode_repetition_3, prepare_logical_state, build_core_circuit_with_syndrome
from utils.logging import (
    apply_x_correction,
    decode_logical_bit_majority,
    decode_repetition_syndrome_block_consensus,
    decode_repetition_syndrome_majority,
    parse_counts_key_default,
    score_logical_success,
    score_raw_majority_success,
)


def _try_aer_simulator(*, noise_model=None, seed: Optional[int] = None):
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator(method="automatic", noise_model=noise_model, seed_simulator=seed)
    except ImportError:
        return None


def _simulator(*, noise_model=None, seed: Optional[int] = None):
    aer = _try_aer_simulator(noise_model=noise_model, seed=seed)
    if aer is not None:
        return aer
    if noise_model is not None:
        raise ImportError("qiskit-aer is required for noisy simulation. Install with: pip install qiskit-aer")
    return BasicSimulator()


def _parse_key_data3_mem1(key: str) -> Tuple[str, str]:
    """
    Parse get_counts() key for a circuit that measures:
    - data register d[0..2] (3 bits)
    - memory register m[0]   (1 bit)

    Qiskit prints registers MSB->LSB and separates registers with spaces.
    We return index-order bits:
        d_bits = d0d1d2
        m_bit  = m0
    """
    parts = key.strip().split()
    if len(parts) == 2:
        a, b = parts[0], parts[1]
        if len(a) == 3 and len(b) == 1:
            d_group, m_group = a, b
        elif len(b) == 3 and len(a) == 1:
            d_group, m_group = b, a
        else:
            bits = (a + b).replace(" ", "")
            if len(bits) < 4:
                raise ValueError(f"Invalid key: {key!r}")
            d_group, m_group = bits[:3], bits[3]
    else:
        bits = key.replace(" ", "")
        if len(bits) < 4:
            raise ValueError(f"Invalid key: {key!r}")
        d_group, m_group = bits[:3], bits[3]

    d_bits = d_group[::-1]
    m_bit = m_group[::-1]  # length 1, but keep consistent
    return d_bits, m_bit


@dataclass(frozen=True)
class ClassicalConsensusParams:
    logical_state: str = "0"
    n_cycles: int = 5
    shots: int = 8000
    seed: int = 11
    block_W: int = 4


def run_classical_consensus(p: ClassicalConsensusParams) -> Dict[str, float]:
    """
    Build the repetition-code syndrome-logging circuit and score:
    - raw majority (no QEC)
    - flat syndrome majority (QEC baseline)
    - block-consensus syndrome decode (blockchain-like)
    """
    core_p = CoreParams(n_cycles=p.n_cycles, idle_ticks_data=0, idle_ticks_anc=0, logical_state=p.logical_state)
    qc = build_core_circuit_with_syndrome(core_p, n_cycles=core_p.n_cycles, measure_data=True)

    sim = _simulator(seed=p.seed)
    result = sim.run(qc, shots=p.shots).result()
    counts = result.get_counts(qc)

    raw = score_raw_majority_success(counts, n_cycles=core_p.n_cycles, ideal_logical=p.logical_state)
    flat = score_logical_success(counts, n_cycles=core_p.n_cycles, ideal_logical=p.logical_state)

    # Block-consensus decode: parse each shot, decode s_bits -> correction target, apply, then majority.
    success = 0
    total = 0
    for key, c in counts.items():
        parsed = parse_counts_key_default(key, core_p.n_cycles)
        target = decode_repetition_syndrome_block_consensus(parsed.s_bits, n_cycles=core_p.n_cycles, W=p.block_W)
        corrected = apply_x_correction(parsed.d_bits, target)
        logical = decode_logical_bit_majority(corrected)
        if logical == p.logical_state:
            success += c
        total += c
    block = success / total if total else 0.0

    return {"raw_majority": raw, "flat_majority": flat, "block_consensus": block}


@dataclass(frozen=True)
class QuantumMemoryParams:
    """
    A minimal "quantum memory consensus" model.

    We keep a single buffer qubit (q3) unmeasured across cycles. Each cycle:
    - apply parity-dependent phase kicks (Z0Z1 and Z1Z2) onto the buffer
    - apply a fixed rotation (RY) to make the update history-dependent
    - optional dynamical-decoupling X pulses on the buffer

    At the end, measure buffer -> m0 (in X basis if measure_in_x=True) and data -> d[0..2].
    """

    logical_state: str = "0"  # recommended: "0" or "1" for simple scoring
    n_cycles: int = 5
    shots: int = 8000
    seed: int = 11

    # Parity kick strengths: phi=pi matches a clean parity phase flip.
    phi_01: float = 3.141592653589793
    phi_12: float = 3.141592653589793

    # Memory update rotation (unconditional).
    mem_ry: float = 0.35

    # Low-frequency noise / detuning proxy on the buffer per cycle: exp(-i * phase/2 * Z).
    # Use together with dd_scheme="echo" to model refocusing.
    buffer_detune_phase: float = 0.0

    # Dynamical decoupling scheme applied to the buffer to suppress Z-noise.
    # - "none": apply buffer_detune_phase directly (accumulates)
    # - "echo": RZ(phase/2) X RZ(phase/2) X  (refocuses Z detuning in ideal sim)
    dd_scheme: str = "none"  # "none" | "echo"

    # Optional "purification": reset memory every K cycles (non-unitary; breaks history).
    purify_every: int = 0

    measure_in_x: bool = True

    # Naive post-process correction: if m=1, flip this data index before majority decode.
    assisted_flip_target: int = 0


def build_quantum_memory_consensus_circuit(
    p: QuantumMemoryParams,
    *,
    hook: Optional[Callable[[QuantumCircuit, int], None]] = None,
) -> QuantumCircuit:
    q = QuantumRegister(4, "q")  # data q0,q1,q2 + buffer q3
    d = ClassicalRegister(3, "d")
    m = ClassicalRegister(1, "m")
    qc = QuantumCircuit(q, m, d)  # key prints as "d m" (reverse reg order) in most Qiskit setups

    prepare_logical_state(qc, p.logical_state)
    encode_repetition_3(qc)

    buf = 3
    # Start buffer in |+> so parity kicks show up as relative phase in X basis.
    qc.h(buf)

    for k in range(int(p.n_cycles)):
        if hook is not None:
            hook(qc, k)

        # Phase kicks encoding Z-stabilizer parities into buffer phase.
        # Z0Z1: apply CRZ(phi_01) from q0 and q1 to buffer.
        qc.crz(float(p.phi_01), 0, buf)
        qc.crz(float(p.phi_01), 1, buf)

        # Z1Z2: apply CRZ(phi_12) from q1 and q2 to buffer.
        qc.crz(float(p.phi_12), 1, buf)
        qc.crz(float(p.phi_12), 2, buf)

        # Unconditional memory update. With interleaved parity kicks, this makes the update
        # depend on the syndrome history (non-commuting updates).
        qc.ry(float(p.mem_ry), buf)

        # Buffer detuning/noise proxy + optional dynamical decoupling.
        if p.buffer_detune_phase:
            phase = float(p.buffer_detune_phase)
            if p.dd_scheme == "echo":
                qc.rz(phase / 2.0, buf)
                qc.x(buf)
                qc.rz(phase / 2.0, buf)
                qc.x(buf)
            elif p.dd_scheme == "none":
                qc.rz(phase, buf)
            else:
                raise ValueError(f"Unknown dd_scheme={p.dd_scheme!r}. Use 'none' or 'echo'.")

        # Optional purification: periodically reset the buffer (breaks history by design).
        if p.purify_every and p.purify_every > 0 and ((k + 1) % int(p.purify_every) == 0):
            qc.reset(buf)
            qc.h(buf)

    # Readout: measure buffer (optionally in X basis) + data.
    if p.measure_in_x:
        qc.h(buf)
    qc.measure(buf, m[0])

    qc.measure(0, d[0])
    qc.measure(1, d[1])
    qc.measure(2, d[2])
    return qc


def score_quantum_memory_counts(counts: Dict[str, int], *, ideal_logical: str, flip_target: int) -> Dict[str, float]:
    """
    Score for the quantum-memory circuit:
    - raw_majority_success: majority vote on data only
    - assisted_success: if memory bit m==1, flip chosen data bit, then majority
    - p_m1: probability that memory bit is 1
    """
    ideal_logical = str(ideal_logical).strip()
    if ideal_logical not in ("0", "1"):
        raise ValueError("ideal_logical must be '0' or '1' for this simple score.")
    if flip_target not in (0, 1, 2):
        raise ValueError("flip_target must be 0, 1, or 2.")

    total = sum(counts.values())
    if total == 0:
        return {"raw_majority_success": 0.0, "assisted_success": 0.0, "p_m1": 0.0}

    raw_ok = 0
    assisted_ok = 0
    m1 = 0

    for key, c in counts.items():
        d_bits, m_bit = _parse_key_data3_mem1(key)

        logical_raw = decode_logical_bit_majority(d_bits)
        if logical_raw == ideal_logical:
            raw_ok += c

        if m_bit == "1":
            m1 += c
            d_corr = apply_x_correction(d_bits, flip_target)
        else:
            d_corr = d_bits
        logical_assisted = decode_logical_bit_majority(d_corr)
        if logical_assisted == ideal_logical:
            assisted_ok += c

    return {
        "raw_majority_success": raw_ok / total,
        "assisted_success": assisted_ok / total,
        "p_m1": m1 / total,
    }


def run_quantum_memory_consensus(p: QuantumMemoryParams) -> Dict[str, float]:
    qc = build_quantum_memory_consensus_circuit(p)
    sim = _simulator(seed=p.seed)
    result = sim.run(qc, shots=p.shots).result()
    counts = result.get_counts(qc)
    return score_quantum_memory_counts(
        counts,
        ideal_logical=p.logical_state,
        flip_target=p.assisted_flip_target,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["classical", "quantum_memory"], required=True)
    ap.add_argument("--logical_state", type=str, default="0")
    ap.add_argument("--n_cycles", type=int, default=5)
    ap.add_argument("--shots", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=11)

    # classical consensus
    ap.add_argument("--block_W", type=int, default=4)

    # quantum memory consensus
    ap.add_argument("--phi_01", type=float, default=3.141592653589793)
    ap.add_argument("--phi_12", type=float, default=3.141592653589793)
    ap.add_argument("--mem_ry", type=float, default=0.35)
    ap.add_argument("--buffer_detune_phase", type=float, default=0.0)
    ap.add_argument("--dd_scheme", choices=["none", "echo"], default="none")
    ap.add_argument("--purify_every", type=int, default=0)
    ap.add_argument("--measure_in_x", action="store_true")
    ap.add_argument("--assisted_flip_target", type=int, default=0, choices=[0, 1, 2])
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    if args.mode == "classical":
        out = run_classical_consensus(
            ClassicalConsensusParams(
                logical_state=args.logical_state,
                n_cycles=args.n_cycles,
                shots=args.shots,
                seed=args.seed,
                block_W=args.block_W,
            )
        )
        print(out)
    else:
        out = run_quantum_memory_consensus(
            QuantumMemoryParams(
                logical_state=args.logical_state,
                n_cycles=args.n_cycles,
                shots=args.shots,
                seed=args.seed,
                phi_01=args.phi_01,
                phi_12=args.phi_12,
                mem_ry=args.mem_ry,
                buffer_detune_phase=args.buffer_detune_phase,
                dd_scheme=args.dd_scheme,
                purify_every=args.purify_every,
                measure_in_x=args.measure_in_x,
                assisted_flip_target=args.assisted_flip_target,
            )
        )
        print(out)


if __name__ == "__main__":
    main()
