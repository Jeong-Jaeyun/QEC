"""
Scenario 2 (MVP): 3큐비트 하의 위상 기하학적 최소 아날로그 (Topological-minimal analogue)

구현 내용:
- 반복적인 Z-패리티 추출: 
    데이터 큐비트(q0, q1)의 Z-패리티를 보조 큐비트(q2)로 반복해서 추출
- 선택적 최소 피드백: 
    만약 증후군(syndrome)이 1 이면, 하나의 데이터 큐비트에 국소적 교정(local correction)을 적용
"""

from dataclasses import dataclass
from qiskit import QuantumCircuit
from core.circuit import ScenarioHook

@dataclass
class TopoMinimalParams:
    # Constraint extraction strength: how many times parity-check pattern per cycle
    check_reps: int = 1

    # If True, uncompute ancilla after encoding parity (reduces unwanted entanglement)
    uncompute: bool = True

    # Minimal feedback (classical feedforward) — implemented as a *coherent* proxy:
    # apply a controlled-X from ancilla to data to mimic "if syndrome==1 then correct".
    # (In real hardware you'd use dynamic circuits; here we keep it simulator-stable.)
    feedback: bool = True

    # Which qubit to "correct" when syndrome=1
    correct_target: int = 0  # 0 or 1

def make_topo_hook(p: TopoMinimalParams) -> ScenarioHook:
    """
    Hook that performs:
    parity encode: CX(0->2), CX(1->2)  (ancilla stores Z-parity)
    optional uncompute: reverse
    optional coherent feedback: CX(2->target)  (proxy for conditional correction)
    """
    assert p.correct_target in (0, 1)

    def hook(qc: QuantumCircuit, k: int) -> None:
        reps = max(1, p.check_reps)
        for _ in range(reps):
            # Encode Z-parity of data onto ancilla
            qc.cx(0, 2)
            qc.cx(1, 2)

            if p.feedback:
                # Coherent proxy: if ancilla=1, flip chosen data qubit
                qc.cx(2, p.correct_target)

            if p.uncompute:
                # Uncompute parity to reduce lingering entanglement
                qc.cx(1, 2)
                qc.cx(0, 2)

    return hook
