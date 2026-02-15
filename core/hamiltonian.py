from __future__ import annotations

import numpy as np
from qiskit.circuit.library import UnitaryGate

# This module contains small, explicit Hamiltonian-to-unitary helpers for toy models.
# They are intentionally limited to <= 3 qubits so we can build exact matrices with numpy.


def zzz_unitary(theta: float) -> UnitaryGate:
    """
    Legacy toy unitary: U = exp(-i * theta * ZZZ).

    Eigenvalue of ZZZ is (+1) for even parity, (-1) for odd parity.
    """
    phases = []
    for b in range(8):
        b0 = (b >> 0) & 1
        b1 = (b >> 1) & 1
        b2 = (b >> 2) & 1
        parity = (b0 + b1 + b2) & 1
        eigen = +1 if parity == 0 else -1
        phases.append(np.exp(-1j * float(theta) * float(eigen)))
    U = np.diag(phases).astype(complex)
    return UnitaryGate(U, label="U_ZZZ")


def _unitary_from_hermitian(H: np.ndarray, t: float) -> np.ndarray:
    """
    Exact unitary exp(-i t H) for a small Hermitian matrix H via eigendecomposition.
    """
    w, v = np.linalg.eigh(H)
    return v @ np.diag(np.exp(-1j * float(t) * w)) @ v.conj().T


def _sigma_plus() -> np.ndarray:
    return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _sigma_minus() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


def _pauli_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _id2() -> np.ndarray:
    return np.eye(2, dtype=complex)


def exchange_hamiltonian_2q(g: float, detuning: float = 0.0) -> np.ndarray:
    """
    Two-qubit exchange Hamiltonian (data, buffer) in the rotating-wave approximation:

        H = g (sigma+_d sigma-_b + sigma-_d sigma+_b) + (detuning/2) * Z_b

    where "buffer" is the SECOND qubit of the 2-qubit ordering.

    In bosonic notation this matches: g (a_d^dagger a_b + a_b^dagger a_d)
    in the single-excitation subspace.
    """
    g = float(g)
    detuning = float(detuning)

    sp = _sigma_plus()
    sm = _sigma_minus()
    Z = _pauli_z()
    I = _id2()

    H_ex = g * (np.kron(sp, sm) + np.kron(sm, sp))
    H_det = 0.5 * detuning * np.kron(I, Z)
    H = H_ex + H_det
    return 0.5 * (H + H.conj().T)


def exchange_unitary_2q(g: float, t: float, detuning: float = 0.0, label: str = "U_EX") -> UnitaryGate:
    """
    Exact unitary evolution exp(-i t H) for exchange_hamiltonian_2q.
    """
    H = exchange_hamiltonian_2q(g=g, detuning=detuning)
    U = _unitary_from_hermitian(H, t=t)
    return UnitaryGate(U, label=label)


def buffer_bus_hamiltonian_3q(g1: float, g2: float, detuning_b: float = 0.0) -> np.ndarray:
    """
    3-qubit "buffer bus" Hamiltonian with no direct data-data coupling.

    Qubit ordering is (d1, b, d2).

        H = g1 (sigma+_d1 sigma-_b + h.c.) + g2 (sigma+_d2 sigma-_b + h.c.)
            + (detuning_b/2) Z_b

    In bosonic notation: sum_i g_i (a_i^dagger a_b + a_b^dagger a_i).

    In the dispersive regime |detuning_b| >> g_i, an effective data-data exchange emerges:
        H_eff ~ (g1*g2/detuning_b) (sigma+_d1 sigma-_d2 + h.c.) + Stark shifts.
    """
    g1 = float(g1)
    g2 = float(g2)
    detuning_b = float(detuning_b)

    sp = _sigma_plus()
    sm = _sigma_minus()
    Z = _pauli_z()
    I = _id2()

    # d1-b coupling: (sp ⊗ sm ⊗ I) + (sm ⊗ sp ⊗ I)
    H_1b = g1 * (np.kron(np.kron(sp, sm), I) + np.kron(np.kron(sm, sp), I))

    # b-d2 coupling: (I ⊗ sp ⊗ sm) + (I ⊗ sm ⊗ sp)
    H_bd2 = g2 * (np.kron(np.kron(I, sp), sm) + np.kron(np.kron(I, sm), sp))

    # detuning on buffer (middle qubit): I ⊗ Z ⊗ I
    H_det = 0.5 * detuning_b * np.kron(np.kron(I, Z), I)

    H = H_1b + H_bd2 + H_det
    return 0.5 * (H + H.conj().T)


def buffer_bus_unitary_3q(
    g1: float,
    g2: float,
    t: float,
    detuning_b: float = 0.0,
    label: str = "U_BUS",
) -> UnitaryGate:
    """
    Exact unitary evolution exp(-i t H) for buffer_bus_hamiltonian_3q.
    """
    H = buffer_bus_hamiltonian_3q(g1=g1, g2=g2, detuning_b=detuning_b)
    U = _unitary_from_hermitian(H, t=t)
    return UnitaryGate(U, label=label)

