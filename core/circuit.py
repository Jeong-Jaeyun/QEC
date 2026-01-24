from dataclasses import dataclass
from typing import Callable, Optional
from qiskit.circuit import Measure
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

def build_core_circuit(
    p: CoreParams,
    hook: Optional[ScenarioHook] = None,
    mode: str = "decoded",   # "state" | "decoded"
    reset_ancilla: bool = True,
) -> QuantumCircuit:
    if mode not in ("state", "decoded"):
        raise ValueError(f"Invalid mode: {mode}. Use 'state' or 'decoded'.")

    measure_ancilla = (mode == "decoded")

    qc = QuantumCircuit(3, p.n_cycles if measure_ancilla else 0)
    prepare_initial_state(qc)

    U = zzz_unitary(p.theta)

    for k in range(p.n_cycles):
        qc.append(U, [0, 1, 2])

        for _ in range(p.idle_ticks_data):
            qc.id(0); qc.id(1)
        for _ in range(p.idle_ticks_anc):
            qc.id(2)

        for _ in range(p.extra_data_ticks):
            qc.id(0); qc.id(1)
        for _ in range(p.extra_anc_ticks):
            qc.id(2)

        if hook is not None:
            hook(qc, k)

        if measure_ancilla:
            qc.measure(2, k)
            if reset_ancilla:
                qc.reset(2)

    # only in state mode
    if getattr(p, "save_density", False) and mode == "state":
        qc.save_density_matrix()

    return qc


def build_core_circuit_with_syndrome(core_p: CoreParams, hook=None, n_cycles=None) -> QuantumCircuit:
    """
    Same core dynamics as build_core_circuit, but:
      - logs ancilla measurement each cycle into s[0..n_cycles-1]
      - measures data qubits at the end into d[0], d[1]
    """
    if n_cycles is None:
        n_cycles = core_p.n_cycles

    qreg = QuantumRegister(3, "q")
    sreg = ClassicalRegister(n_cycles, "s")
    dreg = ClassicalRegister(2, "d")
    qc = QuantumCircuit(qreg, sreg, dreg)

    prepare_initial_state(qc)
    U = zzz_unitary(core_p.theta)

    for k in range(n_cycles):
        qc.append(U, [0, 1, 2])

        for _ in range(core_p.idle_ticks_data):
            qc.id(0); qc.id(1)
        for _ in range(core_p.idle_ticks_anc):
            qc.id(2)

        for _ in range(core_p.extra_data_ticks):
            qc.id(0); qc.id(1)
        for _ in range(core_p.extra_anc_ticks):
            qc.id(2)

        if hook is not None:
            hook(qc, k)

        qc.measure(2, sreg[k])
        qc.reset(2)

    qc.measure(0, dreg[0])
    qc.measure(1, dreg[1])
    return qc
