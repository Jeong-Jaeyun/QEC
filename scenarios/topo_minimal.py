"""
Scenario 2 (MVP): 3큐비트 하의 위상 기하학적 최소 아날로그 (Topological-minimal analogue)

구현 내용:
- 반복적인 Z-패리티 추출:
    데이터 큐비트(q0, q1)의 Z-패리티를 보조 큐비트(q2)로 반복해서 추출
- 선택적 최소 피드백:
    만약 증후군(syndrome)이 1 이면, 하나의 데이터 큐비트에 국소적 교정(local correction)을 적용

이 파일은 시나리오 훅(hook)을 'core.circuit'에서 요구하는 인터페이스로 제공한다.
"""

from dataclasses import dataclass
from typing import Callable, Tuple
from qiskit import QuantumCircuit
from core.circuit import ScenarioHook


Hook = Callable[[QuantumCircuit, int], None]


@dataclass(frozen=True)
class TopoMinimalParams:
    # how many parity-check patterns per cycle
    check_reps: int = 1

    # uncompute ancilla after parity encode
    uncompute: bool = True

    # coherent proxy for conditional correction (dynamic circuit 대체)
    feedback: bool = False

    # which data qubit to flip when syndrome=1
    correct_target: int = 0  # 0 or 1


def make_topo_hook(p: TopoMinimalParams) -> ScenarioHook:
    """
    parity encode: CX(0->2), CX(1->2)
    optional feedback: CX(2->target)
    optional uncompute: CX(1->2), CX(0->2)
    """
    if p.correct_target not in (0, 1):
        raise ValueError(f"correct_target must be 0 or 1, got {p.correct_target}")

    def hook(qc: QuantumCircuit, k: int) -> None:
        reps = max(1, p.check_reps)
        for _ in range(reps):
            qc.cx(0, 2)
            qc.cx(1, 2)

            if p.feedback:
                qc.cx(2, p.correct_target)

            if p.uncompute:
                qc.cx(1, 2)
                qc.cx(0, 2)

    return hook


def make_hooks(args=None) -> Tuple[Hook, Hook, Hook]:
    """
    Returns:
    S2a: constraint only
    S2b: constraint + coherent feedback
    S2_log: logging only (decoded/S3용; feedback 없음, uncompute 없음)
    """

    # S2a: constraint only
    p_a = TopoMinimalParams(
        check_reps=1,
        uncompute=True,
        feedback=False,
        correct_target=0,
    )

    # S2b: constraint + feedback
    p_b = TopoMinimalParams(
        check_reps=1,
        uncompute=True,
        feedback=True,
        correct_target=0,
    )

    # S2_log: syndrome logging only
    # - feedback 없음 (decoder가 밖에서 처리)
    # - uncompute=False로 ancilla에 패리티 흔적을 남겨 로그가 안정적으로 나오게 함
    p_log = TopoMinimalParams(
        check_reps=1,
        uncompute=False,
        feedback=False,
        correct_target=0,
    )

    return (
        make_topo_hook(p=p_a),
        make_topo_hook(p=p_b),
        make_topo_hook(p=p_log),
    )
