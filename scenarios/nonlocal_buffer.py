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
from core.circuit import CoreParams, ScenarioHook
from core.noise import NoiseParams
#from __future__ import annotations


Hook = Callable[[QuantumCircuit, int], None]


@dataclass
class NonLocalBufferParams:
    extra_anc_ticks: int = 3          # extra ancilla id ticks per cycle
    bridge_strength: int = 1          # how many times to apply bridging pattern per cycle
    bridge_mode: str = "zx"           # "zx" or "xx" variants
    enabled: bool = True


def make_nonlocal_buffer_hook(p: NonLocalBufferParams) -> ScenarioHook:
    """
    Returns a hook(qc, k) that adds ancilla-mediated operations each cycle.
    """
    def hook(qc: QuantumCircuit, k: int) -> None:
        if not p.enabled:
            return

        # "remote latency" on ancilla
        for _ in range(max(0, p.extra_anc_ticks)):
            qc.id(2)

        # ancilla-mediated bridging (intentionally no uncompute)
        for _ in range(max(1, p.bridge_strength)):
            if p.bridge_mode == "zx":
                qc.cx(0, 2)
                qc.cz(2, 1)
            elif p.bridge_mode == "xx":
                qc.cx(0, 2)
                qc.cx(2, 1)
            else:
                # fallback
                qc.cx(0, 2)
                qc.cx(2, 1)

    return hook

def make_hook(args=None, params: Optional[NonLocalBufferParams] = None) -> ScenarioHook:
    """
    scenarios/* 공통 엔트리.
    compare.py에서 통일된 방식으로 호출 가능하게 만든다.
    """
    if params is None:
        params = NonLocalBufferParams(enabled=True)
    return make_nonlocal_buffer_hook(p=params)


def override_core_params(core: CoreParams, p: NonLocalBufferParams) -> CoreParams:
    # Keep core immutable: return a modified copy
    return CoreParams(
        theta=core.theta,
        n_cycles=core.n_cycles,
        idle_ticks_data=core.idle_ticks_data,
        idle_ticks_anc=core.idle_ticks_anc,
        extra_anc_ticks=core.extra_anc_ticks + p.extra_anc_ticks,
        extra_data_ticks=core.extra_data_ticks,
    )


def override_noise_params(noise: NoiseParams,
                        anc_strength: float = 1.0,
                        ro_anc: Optional[float] = None,
                        ) -> NoiseParams:
    """
    Increase ancilla noise relative to baseline (represents remote buffer fragility).
    """
    return NoiseParams(
        p1_data=noise.p1_data,
        p1_anc=noise.p1_anc * anc_strength,
        pid_data=noise.pid_data,
        pid_anc=noise.pid_anc * anc_strength,
        ro_anc=(noise.ro_anc if ro_anc is None else ro_anc),
    )
