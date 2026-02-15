"""
Scenario hooks for the 3-qubit repetition (bit-flip) code.

The core circuit (core.circuit.build_core_circuit_with_syndrome) already performs:
1) Encode |psi> -> alpha|000> + beta|111> on data qubits (q0,q1,q2)
2) Repeated stabilizer measurements per cycle:
   - Z0Z1 parity -> syndrome bit s[2k]
   - Z1Z2 parity -> syndrome bit s[2k+1]
   using a single reusable ancilla q3
3) Final data measurement into d[0..2]

This file keeps the older "make_hooks" API used by experiments, but the hooks here
are now focused on injecting simple faults for demonstration/debugging.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env_check import require_typing_extensions_self

require_typing_extensions_self(package="qiskit")

from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate

from core.circuit import ScenarioHook
from core.circuit import encode_repetition_3, prepare_logical_state


Hook = Callable[[QuantumCircuit, int], None]


@dataclass(frozen=True)
class TopoMinimalParams:
    # Inject a single deterministic X error on a chosen data qubit.
    inject_x: bool = False
    inject_target: int = 0  # data qubit index: 0,1,2
    inject_cycle: int = 0   # which syndrome-extraction cycle to inject at


def make_inject_x_hook(p: TopoMinimalParams) -> ScenarioHook:
    if p.inject_target not in (0, 1, 2):
        raise ValueError(f"inject_target must be 0,1,2; got {p.inject_target}")
    if p.inject_cycle < 0:
        raise ValueError(f"inject_cycle must be >= 0; got {p.inject_cycle}")

    def hook(qc: QuantumCircuit, k: int) -> None:
        if not p.inject_x:
            return
        if k == p.inject_cycle:
            qc.x(p.inject_target)

    return hook


def make_hooks(args=None) -> Tuple[Hook, Hook, Hook]:
    """
    Backward-compatible helper returning 3 hooks:
    - S2a: no injection (baseline)
    - S2b: inject a single X error at cycle 0 on data qubit 0
    - log: same as baseline (core already logs syndrome)
    """
    h_a = make_inject_x_hook(TopoMinimalParams(inject_x=False))
    h_b = make_inject_x_hook(TopoMinimalParams(inject_x=True, inject_target=0, inject_cycle=0))
    h_log = make_inject_x_hook(TopoMinimalParams(inject_x=False))
    return (h_a, h_b, h_log)


@dataclass(frozen=True)
class RotatingBufferParams:
    """
    "Rotating buffer" collector (3+1 physical qubits).

    The buffer q3 sequentially couples to data qubits q0->q1->q2, and is measured only once at the end.
    This can create time-delayed correlations that act like a longer "virtual" interaction path.
    """

    logical_state: str = "0"
    n_cycles: int = 5

    # Strong exchange coupling angle for XX+YY interaction (iSWAP-like).
    exchange_theta: float = 1.5707963267948966  # pi/2

    # Optional detuning phase applied to the buffer between couplings.
    detuning_phase: float = 0.0

    # Optional dynamical-decoupling X pulses on the buffer.
    dd_pulses: int = 0

    # Measure buffer in X basis if True.
    measure_buffer_in_x: bool = True


def build_rotating_buffer_circuit(p: RotatingBufferParams) -> QuantumCircuit:
    """
    Circuit layout:
    - q0,q1,q2: data (repetition code block)
    - q3: buffer (collector / bus)

    Classical:
    - c0: buffer measurement
    - d0..d2: final data measurements
    """
    qc = QuantumCircuit(4, 4)

    prepare_logical_state(qc, p.logical_state)
    encode_repetition_3(qc)

    buf = 3

    for _k in range(int(p.n_cycles)):
        # Buffer sequentially couples to each data qubit.
        for data in (0, 1, 2):
            qc.append(XXPlusYYGate(float(p.exchange_theta), 0.0), [data, buf])
            if p.detuning_phase:
                qc.rz(float(p.detuning_phase), buf)
            for _ in range(max(0, int(p.dd_pulses))):
                qc.x(buf)
                qc.id(buf)

    if p.measure_buffer_in_x:
        qc.h(buf)
    qc.measure(buf, 0)

    qc.measure(0, 1)
    qc.measure(1, 2)
    qc.measure(2, 3)
    return qc
