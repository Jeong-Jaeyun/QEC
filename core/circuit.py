from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

# This project previously contained a toy 3-body (ZZZ) dynamics model.
# The current core implements an actual 3-qubit repetition (bit-flip) code:
# - data qubits: q0, q1, q2
# - ancilla:     q3 (reused per stabilizer measurement via reset)
# - stabilizers: Z0Z1 and Z1Z2 (2 syndrome bits per round)


@dataclass
class CoreParams:
    # Legacy field kept for backward compatibility with older scripts/configs.
    theta: float = 0.0

    # Number of repeated syndrome-extraction rounds.
    n_cycles: int = 1

    # Memory-noise approximation: insert 'id' ticks each round.
    idle_ticks_data: int = 1
    idle_ticks_anc: int = 0

    # Initial logical state on data qubit q0 before encoding.
    # Supported: "0", "1", "+", "-"
    logical_state: str = "+"

    # Reset ancilla between stabilizer measurements.
    reset_ancilla: bool = True


ScenarioHook = Callable[[QuantumCircuit, int], None]
# hook signature: hook(qc, cycle_index)


def prepare_logical_state(qc: QuantumCircuit, logical_state: str) -> None:
    """
    Prepare a single-qubit logical state on q0 (data) before encoding.
    """
    s = str(logical_state).strip()
    if s == "0":
        return
    if s == "1":
        qc.x(0)
        return
    if s == "+":
        qc.h(0)
        return
    if s == "-":
        qc.x(0)
        qc.h(0)
        return
    raise ValueError(f"Unsupported logical_state={logical_state!r}. Use one of: '0','1','+','-'.")


def encode_repetition_3(qc: QuantumCircuit) -> None:
    """
    3-qubit repetition (bit-flip) encoding:
        |psi> on q0 -> alpha|000> + beta|111> across q0,q1,q2
    """
    qc.cx(0, 1)
    qc.cx(0, 2)


def _measure_zz_parity(qc: QuantumCircuit, qa: int, qb: int, anc: int, cbit) -> None:
    """
    Measure Z_a Z_b parity using a single ancilla prepared in |0>.
    Using CNOTs into ancilla and measuring ancilla in Z basis yields:
        0 -> even parity (+1 eigenvalue)
        1 -> odd  parity (-1 eigenvalue)
    """
    qc.cx(qa, anc)
    qc.cx(qb, anc)
    qc.measure(anc, cbit)


def build_core_circuit_with_syndrome(
    core_p: CoreParams,
    hook: Optional[ScenarioHook] = None,
    n_cycles: Optional[int] = None,
    *,
    measure_data: bool = True,
) -> QuantumCircuit:
    """
    Build a 3-qubit repetition (bit-flip) code circuit with syndrome logging.

    Quantum registers:
    - q[0:3] data qubits
    - q[3]   ancilla (reused via reset)

    Classical registers:
    - s[0..2*n_cycles-1]  syndrome bits (2 per cycle): [s01, s12] per cycle
    - d[0..2]             final data measurements (optional; set measure_data=False to omit)

    Counts key layout from Qiskit is typically: "d s" (d-register then s-register).
    """
    if n_cycles is None:
        n_cycles = int(core_p.n_cycles)
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    qreg = QuantumRegister(4, "q")
    sreg = ClassicalRegister(2 * n_cycles, "s")
    if measure_data:
        dreg = ClassicalRegister(3, "d")
        qc = QuantumCircuit(qreg, sreg, dreg)
    else:
        qc = QuantumCircuit(qreg, sreg)

    prepare_logical_state(qc, core_p.logical_state)
    encode_repetition_3(qc)

    anc = 3
    for k in range(n_cycles):
        # Idle ticks (memory noise proxy)
        for _ in range(max(0, int(core_p.idle_ticks_data))):
            qc.id(0)
            qc.id(1)
            qc.id(2)
        for _ in range(max(0, int(core_p.idle_ticks_anc))):
            qc.id(anc)

        if hook is not None:
            hook(qc, k)

        # Stabilizer Z0Z1 -> syndrome bit s[2k]
        if core_p.reset_ancilla:
            qc.reset(anc)
        _measure_zz_parity(qc, 0, 1, anc, sreg[2 * k])
        if core_p.reset_ancilla:
            qc.reset(anc)

        # Stabilizer Z1Z2 -> syndrome bit s[2k+1]
        _measure_zz_parity(qc, 1, 2, anc, sreg[2 * k + 1])
        if core_p.reset_ancilla:
            qc.reset(anc)

    # Final data measurement (decoded-mode)
    if measure_data:
        qc.measure(0, dreg[0])
        qc.measure(1, dreg[1])
        qc.measure(2, dreg[2])
    return qc


def build_core_circuit(
    p: CoreParams,
    hook: Optional[ScenarioHook] = None,
) -> QuantumCircuit:
    """
    Convenience wrapper kept for backward compatibility.
    For the repetition code implementation, the main artifact is the syndrome-logging circuit.
    """
    return build_core_circuit_with_syndrome(p, hook=hook, n_cycles=p.n_cycles)
