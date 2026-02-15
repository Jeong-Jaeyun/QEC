from __future__ import annotations

"""
New logic playground for "buffer-only" interactions.

This module is intentionally experimental. It focuses on the physics-motivated ideas:

1) Replace discrete bridge gates (CX/CZ patterns) with a Hamiltonian-level bus model:
      H = sum_i g_i (a_i^dagger a_b + a_b^dagger a_i) + (detuning/2) Z_b
   This produces an effective data-data interaction in the dispersive regime, while keeping
   *no direct* data-data coupling in the physical Hamiltonian.

3) Use high-frequency pi pulses on the buffer as a dynamical-decoupling filter structure.

The code here provides small helpers and demo circuits that run on BasicSimulator (ideal),
and can be extended to Aer/noise models once qiskit-aer is installed.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env_check import require_typing_extensions_self

require_typing_extensions_self(package="qiskit")

from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator

from core.hamiltonian import buffer_bus_unitary_3q, exchange_unitary_2q


def effective_exchange_strength(g1: float, g2: float, detuning_b: float) -> float:
    """
    Second-order (dispersive) estimate of the effective data-data exchange coupling:

        g_eff ~ g1 * g2 / detuning_b

    This is a toy estimate that becomes sensible when |detuning_b| >> max(|g1|,|g2|).
    """
    detuning_b = float(detuning_b)
    if detuning_b == 0.0:
        raise ValueError("detuning_b must be non-zero for dispersive estimate.")
    return float(g1) * float(g2) / detuning_b


def apply_dd_spin_echo(qc: QuantumCircuit, q: int, *, ticks: int = 2) -> None:
    """
    Minimal dynamical decoupling structure:
        id ... X ... id
    In ideal simulation, this is (almost) identity; it matters once you attach a noise model.
    """
    for _ in range(max(0, int(ticks))):
        qc.id(q)
    qc.x(q)
    for _ in range(max(0, int(ticks))):
        qc.id(q)


@dataclass(frozen=True)
class BusTransferParams:
    """
    Demo parameters for a 3-qubit bus unitary applied inside a 4-qubit layout:
    - data qubits: 0,1,2
    - buffer: 3

    We apply the bus unitary on (src, buffer, dst) as ordered qubits.
    """

    src: int = 0
    dst: int = 1
    buffer: int = 3

    g1: float = 0.25
    g2: float = 0.25
    detuning_b: float = 2.0
    t: float = 1.0

    dd: bool = False
    dd_ticks: int = 2

    shots: int = 2000
    seed: int = 11


def build_bus_transfer_circuit(p: BusTransferParams) -> QuantumCircuit:
    qc = QuantumCircuit(4, 4)

    # Prepare a single excitation on src: |1 0 0 0>
    qc.x(p.src)

    # Optional DD structure on buffer before interaction.
    if p.dd:
        apply_dd_spin_echo(qc, p.buffer, ticks=p.dd_ticks)

    U = buffer_bus_unitary_3q(
        g1=p.g1,
        g2=p.g2,
        t=p.t,
        detuning_b=p.detuning_b,
        label="U_BUS",
    )
    qc.append(U, [p.src, p.buffer, p.dst])

    # Measure all qubits for inspection.
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)
    qc.measure(3, 3)
    return qc


def run_bus_transfer_demo(p: BusTransferParams) -> Dict[str, int]:
    qc = build_bus_transfer_circuit(p)
    sim = BasicSimulator()
    res = sim.run(qc, shots=p.shots, seed_simulator=p.seed).result()
    return res.get_counts(qc)


@dataclass(frozen=True)
class SequentialTransferParams:
    """
    Sequential "store and forward" transfer:
      1) exchange(src <-> buffer) for t1
      2) buffer detuning wait (storage) for tau
      3) exchange(buffer <-> dst) for t2
    """

    src: int = 0
    dst: int = 1
    buffer: int = 3

    g1: float = 0.25
    g2: float = 0.25
    t1: float = 1.0
    t2: float = 1.0

    detuning_b: float = 2.0
    tau: float = 1.0

    shots: int = 2000
    seed: int = 11


def build_sequential_transfer_circuit(p: SequentialTransferParams) -> QuantumCircuit:
    qc = QuantumCircuit(4, 4)

    qc.x(p.src)

    # (1) src -> buffer
    qc.append(exchange_unitary_2q(g=p.g1, t=p.t1, detuning=0.0, label="U_EX1"), [p.src, p.buffer])

    # (2) buffer storage (detuning controls "memory time")
    # RZ(lambda) = exp(-i lambda/2 Z), so choose lambda = detuning_b * tau.
    qc.rz(float(p.detuning_b) * float(p.tau), p.buffer)

    # (3) buffer -> dst
    qc.append(exchange_unitary_2q(g=p.g2, t=p.t2, detuning=0.0, label="U_EX2"), [p.dst, p.buffer])

    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)
    qc.measure(3, 3)
    return qc


def run_sequential_transfer_demo(p: SequentialTransferParams) -> Dict[str, int]:
    qc = build_sequential_transfer_circuit(p)
    sim = BasicSimulator()
    res = sim.run(qc, shots=p.shots, seed_simulator=p.seed).result()
    return res.get_counts(qc)
