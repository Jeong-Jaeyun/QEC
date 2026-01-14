from dataclasses import dataclass
from typing import List, Sequence, Tuple
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

@dataclass
class NoiseParams:
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

    # optional: additional "attack" noise on ancilla (implemented via extra id ticks in circuit)
    # but the strength is controlled here via pid_anc or p1_anc, so it actually changes physics.

def build_noise_model(np_: NoiseParams,
                    data_qubits: Sequence[int] = (0,1),
                    anc_qubits: Sequence[int] = (2,),
                    oneq_instr: List[str] = None,
                    twoq_instr: List[str] = None,
                    idle_instr: List[str] = None,
                    data_data_pairs: Sequence[Tuple[int,int]] = ((0,1),),
                    data_anc_pairs: Sequence[Tuple[int,int]] = ((0,2),(1,2))) -> NoiseModel:
    """
    Per-qubit + per-pair noise model.
    - 1Q noise: separate for data vs ancilla.
    - 2Q noise: separate for data-data vs data-ancilla links.
    """
    if oneq_instr is None:
        oneq_instr = ["u1","u2","u3","rx","ry","rz","x","y","z","h","s","sdg","t","tdg"]
    if twoq_instr is None:
        twoq_instr = ["cx", "cz"]
    if idle_instr is None:
        idle_instr = ["id"]

    nm = NoiseModel()

    # 1Q gate noise
    e1_data = depolarizing_error(np_.p1_data, 1)
    e1_anc  = depolarizing_error(np_.p1_anc, 1)
    for q in data_qubits:
        for instr in oneq_instr:
            nm.add_quantum_error(e1_data, instr, [q])
    for q in anc_qubits:
        for instr in oneq_instr:
            nm.add_quantum_error(e1_anc, instr, [q])

    # idle noise
    eid_data = depolarizing_error(np_.pid_data, 1)
    eid_anc  = depolarizing_error(np_.pid_anc, 1)
    for q in data_qubits:
        for instr in idle_instr:
            nm.add_quantum_error(eid_data, instr, [q])
    for q in anc_qubits:
        for instr in idle_instr:
            nm.add_quantum_error(eid_anc, instr, [q])

    # 2Q gate noise (NEW)
    e2_dd = depolarizing_error(np_.p2_data_data, 2)
    e2_da = depolarizing_error(np_.p2_data_anc, 2)

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

    return nm