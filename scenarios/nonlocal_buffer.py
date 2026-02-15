"""
    시나리오 1 (MVP): 비로컬 버퍼 큐비트

    아이디어:
    보조 큐비트를 '원격/비국소(non-local) 버퍼'로 간주
    이로 인해 두 가지 현상이 발생
    1. 추가 지연 및 대기 시간:
        물리적 거리나 통신 과정에서 발생하는 추가적인 레이턴시(Latency)와 유휴 시간(Idle time)이 도입됨
    2. 강한 결합 및 사용:
        보조 큐비트와 데이터 큐비트 사이의 결합을 강화하여,
        보조 큐비트에서 발생한 오류가 데이터 큐비트로 전이(propagate)되도록 만듦
    최소 구현 전략:
    - 보조 큐비트 유휴 틱(Idle ticks) 추가
        원격 통신 시 발생하는 지연 시간이나 네트워크 오버헤드를 시뮬레이션하기 위해 의도적으로 대기 시간을 삽입
    - 반복적인 보조 큐비트 매개 얽힘 경로 (Uncompute 생략)
        보조 큐비트를 이용해 데이터 큐비트들을 얽히게 만드는 과정을 반복하되, 계산을 완전히 되돌리지(uncompute) 않음
        보조 큐비트의 노이즈가 여러 사이클에 걸쳐 데이터 큐비트를 오염 가능
        
    이 코드는 최소화된 구현이며, 추후 확장 예정
"""

from dataclasses import dataclass
from typing import Optional, Callable
from qiskit import QuantumCircuit
from core.circuit import ScenarioHook
from core.hamiltonian import buffer_bus_unitary_3q
#from __future__ import annotations


Hook = Callable[[QuantumCircuit, int], None]


@dataclass
class NonLocalBufferParams:
    extra_anc_ticks: int = 3          # extra ancilla id ticks per cycle
    bridge_strength: int = 1          # how many times to apply bridging pattern per cycle
    bridge_mode: str = "zx"           # "zx" | "xx" | "ham_bus"
    enabled: bool = True

    # Which data qubits to connect via the buffer (q3).
    d_src: int = 0
    d_dst: int = 1

    # Hamiltonian/bus parameters (used when bridge_mode == "ham_bus").
    # Qubit ordering for the unitary is (d_src, buffer, d_dst).
    ham_g1: float = 0.25
    ham_g2: float = 0.25
    ham_detuning: float = 2.0
    ham_time: float = 1.0


def make_nonlocal_buffer_hook(p: NonLocalBufferParams) -> ScenarioHook:
    """
    Returns a hook(qc, k) that adds ancilla-mediated operations each cycle.
    """
    def hook(qc: QuantumCircuit, k: int) -> None:
        if not p.enabled:
            return

        # "remote latency" on ancilla
        for _ in range(max(0, p.extra_anc_ticks)):
            qc.id(3)

        # ancilla-mediated bridging (intentionally no uncompute)
        for _ in range(max(1, p.bridge_strength)):
            if p.bridge_mode == "zx":
                qc.cx(0, 3)
                qc.cz(3, 1)
            elif p.bridge_mode == "xx":
                qc.cx(0, 3)
                qc.cx(3, 1)
            elif p.bridge_mode == "ham_bus":
                # Physical-ish buffer bus model:
                #   H = sum_i g_i (a_i^dagger a_b + a_b^dagger a_i) + (detuning/2) Z_b
                # No direct data-data coupling; energy exchange is mediated only by the buffer.
                U = buffer_bus_unitary_3q(
                    g1=p.ham_g1,
                    g2=p.ham_g2,
                    t=p.ham_time,
                    detuning_b=p.ham_detuning,
                    label="U_BUS",
                )
                qc.append(U, [p.d_src, 3, p.d_dst])
            else:
                # fallback
                qc.cx(0, 3)
                qc.cx(3, 1)

    return hook

def make_hook(args=None, params: Optional[NonLocalBufferParams] = None) -> ScenarioHook:
    """
    scenarios/* 공통 엔트리.
    compare.py에서 통일된 방식으로 호출 가능하게 만든다.
    """
    if params is None:
        params = NonLocalBufferParams(enabled=True)
    return make_nonlocal_buffer_hook(p=params)
