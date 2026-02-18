from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from statistics import mean
from typing import List, Sequence, Tuple

import numpy as np

# Allow running as a script: `python experiments/aer_quasistatic_batch.py ...`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env_check import require_typing_extensions_self

require_typing_extensions_self(package="qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from core.circuit import encode_repetition_3, prepare_logical_state
from core.effective_model import sparse_measurement_rounds
from core.noise import NoiseParams, build_noise_model
from utils.logging import score_logical_success, score_raw_majority_success


@dataclass(frozen=True)
class BatchResult:
    batch_index: int
    delta_i: float
    theta_i: float
    shots_i: int
    success_raw: float
    success_qec: float
    p_i: float


def _simulator(*, noise_model, seed: int):
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator(method="automatic", noise_model=noise_model, seed_simulator=seed)
    except ImportError as exc:
        if noise_model is not None:
            raise SystemExit("qiskit-aer is required for noisy mode. Install with: pip install qiskit-aer") from exc
        from qiskit.providers.basic_provider import BasicSimulator

        return BasicSimulator()


def _measure_zz_parity(qc: QuantumCircuit, qa: int, qb: int, anc: int, cbit) -> None:
    qc.cx(qa, anc)
    qc.cx(qb, anc)
    qc.measure(anc, cbit)


def _measurement_rounds(n_rounds: int, k: int) -> List[int]:
    # Use shared convention from core.effective_model to avoid cross-track drift.
    return list(sparse_measurement_rounds(n_rounds, k))


def build_qsd_template(
    *,
    n_rounds: int,
    k: int,
    logical_state: str,
    theta_param: Parameter,
    idle_ticks_data: int,
    idle_ticks_anc: int,
) -> Tuple[QuantumCircuit, int]:
    rounds_to_measure = _measurement_rounds(n_rounds, k)
    n_measure_rounds = len(rounds_to_measure)

    qreg = QuantumRegister(4, "q")
    sreg = ClassicalRegister(2 * n_measure_rounds, "s")
    dreg = ClassicalRegister(3, "d")
    qc = QuantumCircuit(qreg, sreg, dreg)

    prepare_logical_state(qc, logical_state)
    encode_repetition_3(qc)

    anc = 3
    s_idx = 0
    measure_set = set(rounds_to_measure)
    for r in range(n_rounds):
        for _ in range(max(0, int(idle_ticks_data))):
            qc.id(0)
            qc.id(1)
            qc.id(2)
        for _ in range(max(0, int(idle_ticks_anc))):
            qc.id(anc)

        # Quasi-static drift is inserted once per round, after interaction/idle proxy.
        qc.rz(theta_param, 0)
        qc.rz(theta_param, 1)
        qc.rz(theta_param, 2)

        if r in measure_set:
            qc.reset(anc)
            _measure_zz_parity(qc, 0, 1, anc, sreg[s_idx])
            s_idx += 1

            qc.reset(anc)
            _measure_zz_parity(qc, 1, 2, anc, sreg[s_idx])
            s_idx += 1

            qc.reset(anc)

    qc.measure(0, dreg[0])
    qc.measure(1, dreg[1])
    qc.measure(2, dreg[2])
    return qc, n_measure_rounds


def _bind_theta(qc: QuantumCircuit, theta_param: Parameter, theta_value: float) -> QuantumCircuit:
    if hasattr(qc, "assign_parameters"):
        return qc.assign_parameters({theta_param: theta_value}, inplace=False)
    return qc.bind_parameters({theta_param: theta_value})  # pragma: no cover


def _split_shots(total_shots: int, batches: int) -> List[int]:
    base = total_shots // batches
    rem = total_shots % batches
    return [base + (1 if i < rem else 0) for i in range(batches)]


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    num = sum(float(v) * int(w) for v, w in zip(values, weights))
    den = sum(int(w) for w in weights)
    if den <= 0:
        return 0.0
    return num / float(den)


def _regression_abs_delta(results: Sequence[BatchResult]) -> Tuple[float, float]:
    x = np.asarray([abs(r.delta_i) for r in results], dtype=float)
    y = np.asarray([r.p_i for r in results], dtype=float)
    if len(x) < 2:
        return 0.0, 0.0
    vx = float(np.var(x))
    vy = float(np.var(y))
    if vx <= 0.0:
        return 0.0, 0.0
    cov = float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))))
    slope = cov / vx
    if vy <= 0.0:
        return slope, 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(corr):
        corr = 0.0
    return slope, corr


def parse_args():
    ap = argparse.ArgumentParser(description="AER quasi-static Z-drift batch experiment with sparse monitoring.")
    ap.add_argument("--logical_state", type=str, default="0", choices=["0", "1"])
    ap.add_argument("--n_rounds", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="Sparse monitoring interval. first measurement after k interactions.")
    ap.add_argument("--shots_total", type=int, default=8000)
    ap.add_argument("--batch_M", "--M", dest="batch_M", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)

    ap.add_argument(
        "--sigma_z",
        "--qsd_sigma",
        dest="sigma_z",
        type=float,
        default=0.03,
        help="Quasi-static drift std (Hz scale proxy). Alias: --qsd_sigma",
    )
    ap.add_argument("--tau_int", type=float, default=1.0)
    ap.add_argument("--chi", type=float, default=None, help="Optional metadata for cross-track logging.")
    ap.add_argument("--eta_t", type=float, default=None, help="Optional metadata for cross-track logging.")
    ap.add_argument("--t_m", "--tm", dest="t_m", type=float, default=None, help="Optional measurement latency metadata.")
    ap.add_argument("--t_r", "--tr", dest="t_r", type=float, default=None, help="Optional reset latency metadata.")

    ap.add_argument("--idle_data", type=int, default=1)
    ap.add_argument("--idle_anc", type=int, default=0)

    ap.add_argument("--metric", choices=["qec", "raw"], default="qec", help="Defines P_i = 1 - success_metric.")

    ap.add_argument("--noise", action="store_true", help="Enable qiskit-aer noise model for p_m/p_r and gate noise.")
    ap.add_argument("--channel", choices=["bitflip", "depolarizing"], default="bitflip")
    ap.add_argument("--p1_data", type=float, default=0.0)
    ap.add_argument("--p1_anc", type=float, default=0.0)
    ap.add_argument("--p2_data_data", type=float, default=0.0)
    ap.add_argument("--p2_data_anc", type=float, default=0.0)
    ap.add_argument("--pid_data", type=float, default=0.0)
    ap.add_argument("--pid_anc", type=float, default=0.0)
    ap.add_argument("--p_m", type=float, default=0.0, help="Ancilla readout error probability.")
    ap.add_argument("--p_r", type=float, default=0.0, help="Ancilla reset error probability.")
    ap.add_argument("--ro_data", type=float, default=0.0)
    ap.add_argument("--reset_data", type=float, default=0.0)

    ap.add_argument("--out_prefix", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_rounds <= 0:
        raise ValueError("--n_rounds must be >= 1")
    if args.k <= 0:
        raise ValueError("--k must be >= 1")
    if args.n_rounds < args.k:
        raise ValueError(
            "--n_rounds must be >= --k for sparse syndrome scoring consistency "
            "(first measurement is after k interactions)."
        )
    if args.shots_total <= 0:
        raise ValueError("--shots_total must be >= 1")
    if args.batch_M <= 0:
        raise ValueError("--batch_M must be >= 1")
    if args.sigma_z < 0:
        raise ValueError("--sigma_z must be >= 0")
    if args.tau_int <= 0:
        raise ValueError("--tau_int must be > 0")

    if args.out_prefix is None:
        args.out_prefix = os.path.join(
            "results",
            (
                f"aer_qsd_ls{args.logical_state}_R{args.n_rounds}_k{args.k}"
                f"_M{args.batch_M}_sig{args.sigma_z:g}_pm{args.p_m:g}_pr{args.p_r:g}_seed{args.seed}"
            ),
        )

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    if not args.noise:
        ignored = [
            args.p1_data,
            args.p1_anc,
            args.p2_data_data,
            args.p2_data_anc,
            args.pid_data,
            args.pid_anc,
            args.p_m,
            args.p_r,
            args.ro_data,
            args.reset_data,
        ]
        if any(abs(float(x)) > 0.0 for x in ignored):
            print("warning: noise parameters are ignored unless --noise is set.")

    theta = Parameter("theta_drift")
    template, n_measure_rounds = build_qsd_template(
        n_rounds=args.n_rounds,
        k=args.k,
        logical_state=args.logical_state,
        theta_param=theta,
        idle_ticks_data=args.idle_data,
        idle_ticks_anc=args.idle_anc,
    )

    noise_model = None
    if args.noise:
        noise = NoiseParams(
            channel=args.channel,
            p1_data=args.p1_data,
            p1_anc=args.p1_anc,
            p2_data_data=args.p2_data_data,
            p2_data_anc=args.p2_data_anc,
            pid_data=args.pid_data,
            pid_anc=args.pid_anc,
            ro_anc=args.p_m,
            ro_data=args.ro_data,
            reset_anc=args.p_r,
            reset_data=args.reset_data,
        )
        noise_model = build_noise_model(noise)
    sim = _simulator(noise_model=noise_model, seed=args.seed)

    rng = random.Random(args.seed)
    shots_by_batch = _split_shots(args.shots_total, args.batch_M)
    results: List[BatchResult] = []

    for i, shots_i in enumerate(shots_by_batch):
        if shots_i <= 0:
            continue
        delta_i = rng.gauss(0.0, args.sigma_z)
        theta_i = 2.0 * math.pi * delta_i * args.tau_int
        bound = _bind_theta(template, theta, theta_i)
        run_res = sim.run(bound, shots=shots_i).result()
        counts = run_res.get_counts(bound)

        success_raw = score_raw_majority_success(counts, n_measure_rounds, ideal_logical=args.logical_state)
        success_qec = score_logical_success(counts, n_measure_rounds, ideal_logical=args.logical_state)
        selected_success = success_qec if args.metric == "qec" else success_raw
        p_i = 1.0 - float(selected_success)

        results.append(
            BatchResult(
                batch_index=i,
                delta_i=delta_i,
                theta_i=theta_i,
                shots_i=shots_i,
                success_raw=success_raw,
                success_qec=success_qec,
                p_i=p_i,
            )
        )

        print(
            f"[{i + 1}/{args.batch_M}] shots={shots_i} "
            f"delta={delta_i:.6g} theta={theta_i:.6g} "
            f"raw={success_raw:.6f} qec={success_qec:.6f} P_i={p_i:.6f}"
        )

    if not results:
        raise RuntimeError("No batch executed. Check --shots_total and --batch_M.")

    p_vals = [r.p_i for r in results]
    shot_vals = [r.shots_i for r in results]
    weighted_p = _weighted_mean(p_vals, shot_vals)
    simple_p = float(mean(p_vals))
    slope_abs_delta, corr_abs_delta = _regression_abs_delta(results)
    zeta = args.sigma_z * args.tau_int
    eta_log = args.eta_t
    if eta_log is None and (args.t_m is not None or args.t_r is not None):
        tm = float(args.t_m if args.t_m is not None else 0.0)
        tr = float(args.t_r if args.t_r is not None else 0.0)
        eta_log = (tm + tr) / float(args.tau_int)
    if eta_log is None:
        eta_log = 0.0

    batches_path = f"{args.out_prefix}_batches.csv"
    with open(batches_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "batch_index",
                "delta_i",
                "theta_i",
                "shots_i",
                "success_raw",
                "success_qec",
                "p_i",
            ]
        )
        for row in results:
            w.writerow(
                [
                    row.batch_index,
                    row.delta_i,
                    row.theta_i,
                    row.shots_i,
                    row.success_raw,
                    row.success_qec,
                    row.p_i,
                ]
            )

    summary_path = f"{args.out_prefix}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "logical_state",
                "n_rounds",
                "k",
                "measurement_rounds",
                "shots_total",
                "batch_M",
                "sigma_z",
                "tau_int",
                "zeta",
                "chi",
                "eta_t",
                "t_m",
                "t_r",
                "p_m",
                "p_r",
                "metric",
                "weighted_mean_p",
                "simple_mean_p",
                "weighted_minus_simple",
                "slope_p_vs_abs_delta",
                "corr_p_vs_abs_delta",
            ]
        )
        w.writerow(
            [
                args.logical_state,
                args.n_rounds,
                args.k,
                n_measure_rounds,
                args.shots_total,
                args.batch_M,
                args.sigma_z,
                args.tau_int,
                zeta,
                ("" if args.chi is None else args.chi),
                eta_log,
                ("" if args.t_m is None else args.t_m),
                ("" if args.t_r is None else args.t_r),
                args.p_m,
                args.p_r,
                args.metric,
                weighted_p,
                simple_p,
                weighted_p - simple_p,
                slope_abs_delta,
                corr_abs_delta,
            ]
        )

    print(f"Saved: {batches_path}")
    print(f"Saved: {summary_path}")
    print(
        f"Weighted P={weighted_p:.6f} | Simple P={simple_p:.6f} | "
        f"Slope(P,|delta|)={slope_abs_delta:.6g} | Corr(P,|delta|)={corr_abs_delta:.6g}"
    )


if __name__ == "__main__":
    main()
