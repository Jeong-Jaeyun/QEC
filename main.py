from __future__ import annotations

import argparse
from typing import Optional

from utils.env_check import require_typing_extensions_self

require_typing_extensions_self(package="qiskit")

from core.circuit import CoreParams, build_core_circuit_with_syndrome
from core.metrics import syndrome_stats
from core.noise import NoiseParams, build_noise_model
from utils.logging import score_logical_success, score_raw_majority_success


def _try_make_simulator(*, noise_model=None, seed: Optional[int] = None):
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator(method="automatic", noise_model=noise_model, seed_simulator=seed)
    except ImportError:
        from qiskit.providers.basic_provider import BasicSimulator

        if noise_model is not None:
            raise ImportError(
                "qiskit-aer is required for noisy simulation. Install with: pip install qiskit-aer"
            )
        return BasicSimulator()


def parse_args():
    p = argparse.ArgumentParser(description="3-qubit repetition (bit-flip) code demo")

    p.add_argument("--logical_state", type=str, default="0", help="logical input on q0: 0,1,+,-")
    p.add_argument("--n_cycles", type=int, default=3, help="number of repeated syndrome-extraction rounds")
    p.add_argument("--shots", type=int, default=2000)
    p.add_argument("--seed", type=int, default=11)

    # Deterministic error injection (useful even without Aer noise)
    p.add_argument("--inject_x", action="store_true", help="inject a single X error deterministically")
    p.add_argument("--inject_target", type=int, default=0, choices=[0, 1, 2], help="data qubit index to flip")
    p.add_argument("--inject_cycle", type=int, default=0, help="cycle index to inject the error at")

    # Optional Aer noise model
    p.add_argument("--noise", action="store_true", help="enable Aer noise model (requires qiskit-aer)")
    p.add_argument("--channel", choices=["bitflip", "depolarizing"], default="bitflip")
    p.add_argument("--p1_data", type=float, default=0.001)
    p.add_argument("--p1_anc", type=float, default=0.001)
    p.add_argument("--p2_data_data", type=float, default=0.002)
    p.add_argument("--p2_data_anc", type=float, default=0.005)
    p.add_argument("--pid_data", type=float, default=0.001)
    p.add_argument("--pid_anc", type=float, default=0.001)
    p.add_argument("--ro_anc", type=float, default=0.01)
    p.add_argument("--ro_data", type=float, default=0.0)
    p.add_argument("--reset_anc", type=float, default=0.0, help="reset error probability on ancilla reset")
    p.add_argument("--reset_data", type=float, default=0.0, help="reset error probability on data reset")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.inject_cycle < 0:
        raise ValueError("--inject_cycle must be >= 0")

    def hook(qc, k: int) -> None:
        if args.inject_x and k == args.inject_cycle:
            qc.x(args.inject_target)

    core_p = CoreParams(
        n_cycles=args.n_cycles,
        idle_ticks_data=1,
        idle_ticks_anc=0,
        logical_state=args.logical_state,
    )

    nm = None
    if args.noise:
        noise_p = NoiseParams(
            channel=args.channel,
            p1_data=args.p1_data,
            p1_anc=args.p1_anc,
            p2_data_data=args.p2_data_data,
            p2_data_anc=args.p2_data_anc,
            pid_data=args.pid_data,
            pid_anc=args.pid_anc,
            ro_anc=args.ro_anc,
            ro_data=args.ro_data,
            reset_anc=args.reset_anc,
            reset_data=args.reset_data,
        )
        nm = build_noise_model(noise_p)

    sim = _try_make_simulator(noise_model=nm, seed=args.seed)

    qc = build_core_circuit_with_syndrome(core_p, hook=hook, n_cycles=core_p.n_cycles, measure_data=True)
    result = sim.run(qc, shots=args.shots).result()
    counts = result.get_counts(qc)

    print("counts (top 10):")
    for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {k}: {v}")

    if args.logical_state in ("0", "1"):
        raw = score_raw_majority_success(counts, core_p.n_cycles, ideal_logical=args.logical_state)
        qec = score_logical_success(counts, core_p.n_cycles, ideal_logical=args.logical_state)
        print(f"raw_majority_success:  {raw:.6f}")
        print(f"syndrome_decode_success: {qec:.6f}")
    else:
        print("logical_state is not Z-basis; decoded success metric is skipped (use --logical_state 0/1).")

    stats = syndrome_stats(counts, core_p.n_cycles)
    print(f"syndrome detection_rate: {stats['detection_rate']:.4f}")
    print(f"syndrome false_negative: {stats['false_negative']:.4f}")


if __name__ == "__main__":
    main()
