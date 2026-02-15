from __future__ import annotations

import argparse
import csv
import os
import sys

# Allow running as a script: `python experiments/compare.py ...`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env_check import require_typing_extensions_self

require_typing_extensions_self(package="qiskit")

from core.circuit import CoreParams, build_core_circuit_with_syndrome
from core.metrics import data_fidelity_from_density_matrix, syndrome_stats
from core.noise import NoiseParams, build_noise_model
from utils.logging import score_logical_success, score_raw_majority_success
from typing import Optional

from utils.plotting import plot_lines_logx


def _try_make_simulator(*, method: str, noise_model=None, seed: Optional[int] = None):
    """
    Prefer Aer (supports noise + density matrix). Fall back to BasicSimulator for ideal sampled runs.
    """
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator(method=method, noise_model=noise_model, seed_simulator=seed)
    except ImportError:
        from qiskit.providers.basic_provider import BasicSimulator

        if method != "automatic" or noise_model is not None:
            raise ImportError(
                "qiskit-aer is required for noisy simulation or density-matrix mode. "
                "Install with: pip install qiskit-aer"
            )
        return BasicSimulator()


def _make_noise(base: NoiseParams, link_mult: float, inject_link: bool = True) -> NoiseParams:
    # For repetition code, the main "link" is data-ancilla CNOTs used in stabilizer measurement.
    return NoiseParams(
        channel=base.channel,
        p1_data=base.p1_data,
        p1_anc=base.p1_anc,
        p2_data_data=base.p2_data_data,
        p2_data_anc=(min(0.95, base.p2_data_anc * link_mult) if inject_link else 0.0),
        pid_data=base.pid_data,
        pid_anc=base.pid_anc,
        ro_anc=base.ro_anc,
        ro_data=base.ro_data,
    )


def run_decoded_sweep(args) -> None:
    os.makedirs("results", exist_ok=True)

    if args.logical_state not in ("0", "1"):
        raise ValueError("--logical_state must be '0' or '1' for decoded sweep.")

    core_p = CoreParams(
        n_cycles=args.n_cycles,
        idle_ticks_data=args.idle_data,
        idle_ticks_anc=args.idle_anc,
        logical_state=args.logical_state,
    )

    base_noise = NoiseParams(
        channel=args.channel,
        p1_data=args.p1_data,
        p1_anc=args.p1_anc,
        p2_data_data=args.p2_data_data,
        p2_data_anc=args.p2_data_anc,
        pid_data=args.pid_data,
        pid_anc=args.pid_anc,
        ro_anc=args.ro_anc,
        ro_data=args.ro_data,
    )

    xs = args.link_mults
    ys = {"raw_majority": [], "syndrome_decode": []}
    ms = {"detection": [], "false_negative": []}

    for lm in xs:
        nm = None
        if args.noise:
            noise = _make_noise(base_noise, lm, inject_link=True)
            nm = build_noise_model(noise)

        sim = _try_make_simulator(method="automatic", noise_model=nm, seed=args.seed)

        qc = build_core_circuit_with_syndrome(core_p, n_cycles=core_p.n_cycles, measure_data=True)
        result = sim.run(qc, shots=args.shots).result()
        counts = result.get_counts(qc)

        ys["raw_majority"].append(score_raw_majority_success(counts, core_p.n_cycles, ideal_logical=args.logical_state))
        ys["syndrome_decode"].append(score_logical_success(counts, core_p.n_cycles, ideal_logical=args.logical_state))

        stats = syndrome_stats(counts, core_p.n_cycles)
        ms["detection"].append(stats["detection_rate"])
        ms["false_negative"].append(stats["false_negative"])

        print(
            f"[decoded] link_mult={lm} raw={ys['raw_majority'][-1]:.6f} "
            f"qec={ys['syndrome_decode'][-1]:.6f} det={ms['detection'][-1]:.4f} fn={ms['false_negative'][-1]:.4f}"
        )

    out_csv = os.path.join("results", "repetition_decoded_sweep.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult", "raw_majority", "syndrome_decode", "detection", "false_negative"])
        for i, lm in enumerate(xs):
            w.writerow([lm, ys["raw_majority"][i], ys["syndrome_decode"][i], ms["detection"][i], ms["false_negative"][i]])
    print(f"Saved: {out_csv}")

    out_png = os.path.join("results", "repetition_decoded_sweep.png")
    plot_lines_logx(
        xs,
        ys,
        title="3-qubit repetition code: decoded success (raw vs syndrome-based)",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="success probability",
        out_png=out_png,
    )
    print(f"Saved: {out_png}")

    out_png2 = os.path.join("results", "repetition_syndrome_stats.png")
    plot_lines_logx(
        xs,
        ms,
        title="Syndrome observability stats",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="rate",
        out_png=out_png2,
    )
    print(f"Saved: {out_png2}")


def run_state_sweep(args) -> None:
    os.makedirs("results", exist_ok=True)

    # state mode requires Aer (density_matrix + save_density_matrix)
    core_p = CoreParams(
        n_cycles=args.n_cycles,
        idle_ticks_data=args.idle_data,
        idle_ticks_anc=args.idle_anc,
        logical_state=args.logical_state,
    )

    base_noise = NoiseParams(
        channel=args.channel,
        p1_data=args.p1_data,
        p1_anc=args.p1_anc,
        p2_data_data=args.p2_data_data,
        p2_data_anc=args.p2_data_anc,
        pid_data=args.pid_data,
        pid_anc=args.pid_anc,
        ro_anc=args.ro_anc,
        ro_data=args.ro_data,
    )

    xs = args.link_mults
    ys = {"data_fidelity": []}

    for lm in xs:
        nm = None
        if args.noise:
            noise = _make_noise(base_noise, lm, inject_link=True)
            nm = build_noise_model(noise)

        sim = _try_make_simulator(method="density_matrix", noise_model=nm, seed=args.seed)

        qc = build_core_circuit_with_syndrome(core_p, n_cycles=core_p.n_cycles, measure_data=False)
        qc.save_density_matrix()
        res = sim.run(qc, shots=1).result()
        from qiskit.quantum_info import DensityMatrix

        rho_full = DensityMatrix(res.data(0)["density_matrix"])

        fid = data_fidelity_from_density_matrix(
            rho_full,
            data_qubits=(0, 1, 2),
            logical_state=args.logical_state,
        )
        ys["data_fidelity"].append(fid)
        print(f"[state] link_mult={lm} fidelity={fid:.6f}")

    out_csv = os.path.join("results", "repetition_state_sweep.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult", "data_fidelity"])
        for i, lm in enumerate(xs):
            w.writerow([lm, ys["data_fidelity"][i]])
    print(f"Saved: {out_csv}")

    out_png = os.path.join("results", "repetition_state_sweep.png")
    plot_lines_logx(
        xs,
        ys,
        title="3-qubit repetition code: data state fidelity",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="fidelity",
        out_png=out_png,
    )
    print(f"Saved: {out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["decoded", "state"], required=True)

    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--shots", type=int, default=8000)

    # repetition-code params
    p.add_argument("--logical_state", type=str, default="0", help="logical input on q0: 0,1,+,-")
    p.add_argument("--n_cycles", type=int, default=5)
    p.add_argument("--idle_data", type=int, default=1)
    p.add_argument("--idle_anc", type=int, default=0)

    # noise params (Aer only)
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

    # sweep
    p.add_argument("--link_mults", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "decoded":
        run_decoded_sweep(args)
    else:
        run_state_sweep(args)


if __name__ == "__main__":
    main()
