import numpy as np
from qiskit.circuit.library import UnitaryGate

def zzz_unitary(theta: float) -> UnitaryGate:
    """
    U = exp(-i * theta * Z⊗Z⊗Z)
    Eigenvalue of ZZZ is (+1) for even parity, (-1) for odd parity.
    """
    phases = []
    for b in range(8):
        b0 = (b >> 0) & 1
        b1 = (b >> 1) & 1
        b2 = (b >> 2) & 1
        parity = (b0 + b1 + b2) & 1
        eigen = +1 if parity == 0 else -1
        phases.append(np.exp(-1j * theta * eigen))
    U = np.diag(phases).astype(complex)
    return UnitaryGate(U, label="U_ZZZ")
