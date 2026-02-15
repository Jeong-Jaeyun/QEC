from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    # Optional dependency. We keep the rest of the repo importable even without Aer.
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

    _AER_AVAILABLE = True
except ImportError:  # pragma: no cover
    NoiseModel = object  # type: ignore[assignment]
    ReadoutError = object  # type: ignore[assignment]
    depolarizing_error = None  # type: ignore[assignment]
    pauli_error = None  # type: ignore[assignment]
    _AER_AVAILABLE = False

@dataclass
class NoiseParams:
    # Noise channel to use for gate/idle errors.
    # - "bitflip": X-only noise (aligned with the 3-qubit repetition code)
    # - "depolarizing": generic depolarizing noise
    channel: str = "bitflip"

    # 1Q depolarizing for gates on data/anc
    p1_data: float = 0.001
    p1_anc: float  = 0.001
    
    # 2Q depolarizing
    p2_data_data: float = 0.002      # for (0,1) if used
    p2_data_anc: float  = 0.005      # for (0,2) and (1,2) links
    
    # idle noise implemented on 'id' gate
    pid_data: float = 0.001
    pid_anc: float  = 0.001

    # measurement readout error (bit-flip on readout)
    # ro_data applies if you ever measure data; ro_anc for ancilla syndrome
    ro_anc: float = 0.01
    ro_data: float = 0.0

    # optional: additional "attack" noise on ancilla (implemented via extra id ticks in circuit)
    # but the strength is controlled here via pid_anc or p1_anc, so it actually changes physics.

def build_noise_model(np_: NoiseParams,
                    data_qubits: Sequence[int] = (0, 1, 2),
                    anc_qubits: Sequence[int] = (3,),
                    oneq_instr: List[str] = None,
                    twoq_instr: List[str] = None,
                    idle_instr: List[str] = None,
                    data_data_pairs: Sequence[Tuple[int,int]] = ((0, 1), (0, 2)),
                    data_anc_pairs: Sequence[Tuple[int,int]] = ((0, 3), (1, 3), (2, 3))) -> NoiseModel:
    """
    Per-qubit + per-pair noise model.
    - 1Q noise: separate for data vs ancilla.
    - 2Q noise: separate for data-data vs data-ancilla links.
    """
    if not _AER_AVAILABLE:  # pragma: no cover
        raise ImportError("qiskit-aer is required for noisy simulation. Install with: pip install qiskit-aer")

    if oneq_instr is None:
        oneq_instr = ["u1","u2","u3","rx","ry","rz","x","y","z","h","s","sdg","t","tdg"]
    if twoq_instr is None:
        twoq_instr = ["cx", "cz"]
    if idle_instr is None:
        idle_instr = ["id"]

    nm = NoiseModel()

    # 1Q gate noise
    if np_.channel == "bitflip":
        e1_data = pauli_error([("X", float(np_.p1_data)), ("I", 1.0 - float(np_.p1_data))])
        e1_anc = pauli_error([("X", float(np_.p1_anc)), ("I", 1.0 - float(np_.p1_anc))])
    elif np_.channel == "depolarizing":
        e1_data = depolarizing_error(float(np_.p1_data), 1)
        e1_anc = depolarizing_error(float(np_.p1_anc), 1)
    else:
        raise ValueError(f"Unknown noise channel: {np_.channel!r}. Use 'bitflip' or 'depolarizing'.")
    for q in data_qubits:
        for instr in oneq_instr:
            nm.add_quantum_error(e1_data, instr, [q])
    for q in anc_qubits:
        for instr in oneq_instr:
            nm.add_quantum_error(e1_anc, instr, [q])

    # idle noise
    if np_.channel == "bitflip":
        eid_data = pauli_error([("X", float(np_.pid_data)), ("I", 1.0 - float(np_.pid_data))])
        eid_anc = pauli_error([("X", float(np_.pid_anc)), ("I", 1.0 - float(np_.pid_anc))])
    else:
        eid_data = depolarizing_error(float(np_.pid_data), 1)
        eid_anc = depolarizing_error(float(np_.pid_anc), 1)
    for q in data_qubits:
        for instr in idle_instr:
            nm.add_quantum_error(eid_data, instr, [q])
    for q in anc_qubits:
        for instr in idle_instr:
            nm.add_quantum_error(eid_anc, instr, [q])

    # 2Q gate noise (NEW)
    if np_.channel == "bitflip":
        # Independent X errors on both qubits.
        e2_dd = pauli_error([("X", float(np_.p2_data_data)), ("I", 1.0 - float(np_.p2_data_data))]).tensor(
            pauli_error([("X", float(np_.p2_data_data)), ("I", 1.0 - float(np_.p2_data_data))])
        )
        e2_da = pauli_error([("X", float(np_.p2_data_anc)), ("I", 1.0 - float(np_.p2_data_anc))]).tensor(
            pauli_error([("X", float(np_.p2_data_anc)), ("I", 1.0 - float(np_.p2_data_anc))])
        )
    else:
        e2_dd = depolarizing_error(float(np_.p2_data_data), 2)
        e2_da = depolarizing_error(float(np_.p2_data_anc), 2)

    # data-data pairs
    for (a,b) in data_data_pairs:
        for instr in twoq_instr:
            nm.add_quantum_error(e2_dd, instr, [a,b])

    # data-ancilla pairs
    for (a,b) in data_anc_pairs:
        for instr in twoq_instr:
            nm.add_quantum_error(e2_da, instr, [a,b])

    # readout error on ancilla
    if np_.ro_anc and np_.ro_anc > 0:
        p = float(np_.ro_anc)
        ro = ReadoutError([[1-p, p],[p, 1-p]])
        for q in anc_qubits:
            nm.add_readout_error(ro, [q])

    # optional readout error on data
    if np_.ro_data and np_.ro_data > 0:
        p = float(np_.ro_data)
        ro = ReadoutError([[1 - p, p], [p, 1 - p]])
        for q in data_qubits:
            nm.add_readout_error(ro, [q])

    return nm
