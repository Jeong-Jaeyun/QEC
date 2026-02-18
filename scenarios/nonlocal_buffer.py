from __future__ import annotations

"""
Scenario 1: non-local buffer mediation hooks.

This module is intentionally separate from the effective-model track. It is a circuit-level
playground for ancilla/buffer-mediated couplings. The key requirement is that data endpoints
must be configurable through d_src/d_dst (no hard-coded (0,1) path).
"""

from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit

from core.circuit import ScenarioHook
from core.hamiltonian import buffer_bus_unitary_3q


@dataclass
class NonLocalBufferParams:
    extra_anc_ticks: int = 3
    bridge_strength: int = 1
    bridge_mode: str = "zx"  # "zx" | "xx" | "ham_bus"
    enabled: bool = True

    # Configurable data endpoints connected through the buffer q3.
    d_src: int = 0
    d_dst: int = 1

    # Hamiltonian bus mode parameters.
    ham_g1: float = 0.25
    ham_g2: float = 0.25
    ham_detuning: float = 2.0
    ham_time: float = 1.0


def _validate_data_idx(idx: int) -> None:
    if idx not in (0, 1, 2):
        raise ValueError(f"Data qubit index must be one of 0,1,2. Got {idx}.")


def make_nonlocal_buffer_hook(p: NonLocalBufferParams) -> ScenarioHook:
    _validate_data_idx(int(p.d_src))
    _validate_data_idx(int(p.d_dst))
    if int(p.d_src) == int(p.d_dst):
        raise ValueError("d_src and d_dst must be different data qubits.")

    def hook(qc: QuantumCircuit, _k: int) -> None:
        if not p.enabled:
            return

        buf = 3
        for _ in range(max(0, int(p.extra_anc_ticks))):
            qc.id(buf)

        for _ in range(max(1, int(p.bridge_strength))):
            if p.bridge_mode == "zx":
                qc.cx(int(p.d_src), buf)
                qc.cz(buf, int(p.d_dst))
            elif p.bridge_mode == "xx":
                qc.cx(int(p.d_src), buf)
                qc.cx(buf, int(p.d_dst))
            elif p.bridge_mode == "ham_bus":
                U = buffer_bus_unitary_3q(
                    g1=float(p.ham_g1),
                    g2=float(p.ham_g2),
                    t=float(p.ham_time),
                    detuning_b=float(p.ham_detuning),
                    label="U_BUS",
                )
                qc.append(U, [int(p.d_src), buf, int(p.d_dst)])
            else:
                raise ValueError(f"Unknown bridge_mode={p.bridge_mode!r}. Use 'zx', 'xx', or 'ham_bus'.")

    return hook


def make_hook(args=None, params: Optional[NonLocalBufferParams] = None) -> ScenarioHook:
    if params is None:
        params = NonLocalBufferParams(enabled=True)
    return make_nonlocal_buffer_hook(p=params)

