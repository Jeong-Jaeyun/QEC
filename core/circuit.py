from dataclasses import dataclass
from typing import Callable, Optional
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from .hamiltonian import zzz_unitary

@dataclass
class CoreParams:
    theta: float = 0.20
    n_cycles: int = 20

    # extra idle ticks per cycle (baseline)
    idle_ticks_data: int = 1
    idle_ticks_anc: int  = 1

    # optional extra ticks (used by scenarios, but core supports it generically)
    extra_anc_ticks: int = 0
    extra_data_ticks: int = 0

def prepare_initial_state(qc: QuantumCircuit) -> None:
    """
    Default initial state for data qubits: Bell-like |Φ+>.
    You can swap later; keep core deterministic.
    """
    qc.h(0)
    qc.cx(0, 1)
    # ancilla in |0>

ScenarioHook = Callable[[QuantumCircuit, int], None]
# hook signature: hook(qc, cycle_index)

def build_core_circuit(p: CoreParams,
                    hook: Optional[ScenarioHook] = None,
                    measure_ancilla: bool = True,
                    reset_ancilla: bool = True) -> QuantumCircuit:
    """
    Build a 3-qubit experiment circuit with:
    prepare -> repeat [ U_ZZZ -> idle -> hook -> measure ancilla -> reset ancilla ]
    Classical bits: n_cycles (store ancilla measurement each cycle).
    """
    qc = QuantumCircuit(3, p.n_cycles if measure_ancilla else 0)
    prepare_initial_state(qc)

    U = zzz_unitary(p.theta)

    for k in range(p.n_cycles):
        qc.append(U, [0,1,2])

        # idle ticks baseline
        for _ in range(p.idle_ticks_data):
            qc.id(0); qc.id(1)
        for _ in range(p.idle_ticks_anc):
            qc.id(2)

        # extra ticks (generic knobs)
        for _ in range(p.extra_data_ticks):
            qc.id(0); qc.id(1)
        for _ in range(p.extra_anc_ticks):
            qc.id(2)

        # scenario hook (optional)
        if hook is not None:
            hook(qc, k)

        if measure_ancilla:
            qc.measure(2, k)
        if reset_ancilla:
            qc.reset(2)

    return qc

def build_core_circuit_with_syndrome(core_p, hook=None, n_cycles=None):
    """
    Build circuit that records ancilla measurement each cycle into a classical syndrome register.
    Also measures data at the end.

    Returns: QuantumCircuit with:
        - quantum regs: 3 qubits
        - classical regs:
          * s[0..n_cycles-1] : ancilla syndrome bits
          * d0, d1 : final data bits
    """
    if n_cycles is None:
        n_cycles = core_p.n_cycles

    qreg = QuantumRegister(3, "q")
    sreg = ClassicalRegister(n_cycles, "s")
    dreg = ClassicalRegister(2, "d")
    
    qc = QuantumCircuit(qreg, sreg, dreg)

    # (초기 상태 준비는 기존 core와 동일하게 유지)
    # 예: 데이터에 H or 준비 상태가 있다면 여기 그대로
    # qc.h(0); qc.h(1) ... 등

    for k in range(n_cycles):
        if hook is not None:
            hook(qc, k)

        qc.measure(2, sreg[k])
        qc.reset(2)

    # 마지막에 데이터 측정
    qc.measure(0, dreg[0])
    qc.measure(1, dreg[1])
    return qc