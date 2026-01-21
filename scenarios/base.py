from typing import Callable
from qiskit import QuantumCircuit
from core.circuit import ScenarioHook

Hook = Callable[[QuantumCircuit, int], None]

def make_hook(args=None) -> ScenarioHook:
    def _hook(qc: QuantumCircuit, cycle_idx: int) -> None:
        return
    return _hook
