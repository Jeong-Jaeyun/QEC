from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.effective_model import (
    MODE_AUTO,
    MODE_FULL,
    MODE_NAIVE,
    EffectiveModelParams,
    sparse_measurement_count,
    simulate_lifetime,
    total_time_for_rounds,
)

_MODE_CHOICES = (MODE_NAIVE, MODE_AUTO, MODE_FULL)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run AER quasi-static drift batch and effective-model simulation with a shared "
            "sparse-monitoring schedule and matched parameter set."
        )
    )
    ap.add_argument("--logical_state", type=str, default="0", choices=["0", "1"])
    ap.add_argument("--n_rounds", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="Sparse monitoring period (first measurement after k rounds).")
    ap.add_argument("--seed", type=int, default=11)

    ap.add_argument("--sigma_z", type=float, default=0.03)
    ap.add_argument("--tau_int", type=float, default=1.0)
    ap.add_argument("--eta_t", type=float, default=1.0)
    ap.add_argument("--t_m", "--tm", dest="t_m", type=float, default=None)
    ap.add_argument("--t_r", "--tr", dest="t_r", type=float, default=None)
    ap.add_argument("--p_m", type=float, default=0.01)
    ap.add_argument("--p_r", type=float, default=0.01)

    ap.add_argument("--gamma_x", type=float, default=0.03)
    ap.add_argument("--chi", type=float, default=4.0)
    ap.add_argument("--mode", choices=_MODE_CHOICES, default=MODE_NAIVE)
    ap.add_argument("--shots_effective", "--shots", dest="shots_effective", type=int, default=1200)
    ap.add_argument("--max_rounds", dest="max_rounds", type=int, default=None, help="Alias for n_rounds in effective track.")

    ap.add_argument("--shots_total", type=int, default=8000, help="Total AER shots across batches.")
    ap.add_argument("--batch_M", "--M", dest="batch_M", type=int, default=32)
    ap.add_argument("--aer_metric", choices=["qec", "raw"], default="qec")
    ap.add_argument("--aer_noise", dest="aer_noise", action="store_true", default=True)
    ap.add_argument("--no_aer_noise", dest="aer_noise", action="store_false")
    ap.add_argument("--channel", choices=["bitflip", "depolarizing"], default="bitflip")
    ap.add_argument("--p1_data", type=float, default=0.0)
    ap.add_argument("--p1_anc", type=float, default=0.0)
    ap.add_argument("--p2_data_data", type=float, default=0.0)
    ap.add_argument("--p2_data_anc", type=float, default=0.0)
    ap.add_argument("--pid_data", type=float, default=0.0)
    ap.add_argument("--pid_anc", type=float, default=0.0)
    ap.add_argument("--ro_data", type=float, default=0.0)
    ap.add_argument("--reset_data", type=float, default=0.0)

    ap.add_argument("--idle_data", type=int, default=1)
    ap.add_argument("--idle_anc", type=int, default=0)

    ap.add_argument("--out_prefix", type=str, default=None)
    return ap.parse_args()


def _resolve_eta(args: argparse.Namespace) -> float:
    if args.t_m is not None or args.t_r is not None:
        tm = float(args.t_m if args.t_m is not None else 0.0)
        tr = float(args.t_r if args.t_r is not None else 0.0)
        return (tm + tr) / float(args.tau_int)
    return float(args.eta_t)


def _aer_command(args: argparse.Namespace, *, eta_t: float, out_prefix: str) -> List[str]:
    cmd = [
        sys.executable,
        os.path.join(_ROOT, "experiments", "aer_quasistatic_batch.py"),
        "--logical_state",
        str(args.logical_state),
        "--n_rounds",
        str(args.n_rounds),
        "--k",
        str(args.k),
        "--shots_total",
        str(args.shots_total),
        "--batch_M",
        str(args.batch_M),
        "--seed",
        str(args.seed),
        "--sigma_z",
        str(args.sigma_z),
        "--tau_int",
        str(args.tau_int),
        "--chi",
        str(args.chi),
        "--eta_t",
        str(eta_t),
        "--p_m",
        str(args.p_m),
        "--p_r",
        str(args.p_r),
        "--metric",
        str(args.aer_metric),
        "--idle_data",
        str(args.idle_data),
        "--idle_anc",
        str(args.idle_anc),
        "--out_prefix",
        str(out_prefix),
    ]
    if args.t_m is not None:
        cmd.extend(["--t_m", str(args.t_m)])
    if args.t_r is not None:
        cmd.extend(["--t_r", str(args.t_r)])
    if args.aer_noise:
        cmd.extend(
            [
                "--noise",
                "--channel",
                str(args.channel),
                "--p1_data",
                str(args.p1_data),
                "--p1_anc",
                str(args.p1_anc),
                "--p2_data_data",
                str(args.p2_data_data),
                "--p2_data_anc",
                str(args.p2_data_anc),
                "--pid_data",
                str(args.pid_data),
                "--pid_anc",
                str(args.pid_anc),
                "--ro_data",
                str(args.ro_data),
                "--reset_data",
                str(args.reset_data),
            ]
        )
    return cmd


def _read_first_csv_row(path: str) -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    raise RuntimeError(f"CSV has no data rows: {path}")


def _run_aer(args: argparse.Namespace, *, eta_t: float, out_prefix: str) -> Dict[str, str]:
    cmd = _aer_command(args, eta_t=eta_t, out_prefix=out_prefix)
    try:
        subprocess.run(cmd, cwd=_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "AER track failed. Ensure qiskit and qiskit-aer are installed in this environment.\n"
            f"Command: {' '.join(cmd)}"
        ) from exc
    summary_path = f"{out_prefix}_summary.csv"
    if not os.path.exists(summary_path):
        raise RuntimeError(f"AER summary file not found: {summary_path}")
    return _read_first_csv_row(summary_path)


def _run_effective(args: argparse.Namespace, *, eta_t: float):
    max_rounds = int(args.max_rounds if args.max_rounds is not None else args.n_rounds)
    zeta = float(args.sigma_z) * float(args.tau_int)
    params = EffectiveModelParams(
        gamma_x=float(args.gamma_x),
        chi=float(args.chi),
        zeta=zeta,
        sigma_z=float(args.sigma_z),
        eta_t=float(eta_t),
        t_m=args.t_m,
        t_r=args.t_r,
        tau_int=float(args.tau_int),
        p_m=float(args.p_m),
        p_r=float(args.p_r),
        k=int(args.k),
        max_rounds=max_rounds,
        shots=int(args.shots_effective),
        seed=int(args.seed),
    )
    stats = simulate_lifetime(params, str(args.mode))
    return params, stats


def _default_prefix(args: argparse.Namespace) -> str:
    return os.path.join(
        "results",
        (
            f"compare_tracks_ls{args.logical_state}_R{args.n_rounds}_k{args.k}"
            f"_sig{args.sigma_z:g}_pm{args.p_m:g}_pr{args.p_r:g}_seed{args.seed}"
        ),
    )


def _write_compare_csv(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.n_rounds <= 0:
        raise ValueError("--n_rounds must be >= 1")
    if args.k <= 0:
        raise ValueError("--k must be >= 1")
    if args.n_rounds < args.k:
        raise ValueError("--n_rounds must be >= --k to include at least one sparse measurement.")
    if args.shots_effective <= 0:
        raise ValueError("--shots_effective must be >= 1")
    if args.shots_total <= 0:
        raise ValueError("--shots_total must be >= 1")
    if args.batch_M <= 0:
        raise ValueError("--batch_M must be >= 1")

    out_prefix = args.out_prefix if args.out_prefix else _default_prefix(args)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    eta_t = _resolve_eta(args)
    aer_prefix = f"{out_prefix}_aer"
    aer_summary = _run_aer(args, eta_t=eta_t, out_prefix=aer_prefix)
    params, stats = _run_effective(args, eta_t=eta_t)

    effective_k = int(stats.k)
    effective_meas_rounds = sparse_measurement_count(int(params.max_rounds), effective_k)
    effective_horizon_time = total_time_for_rounds(
        int(params.max_rounds),
        effective_k,
        float(params.tau_int),
        float(params.eta_t),
    )

    aer_meas_rounds = int(float(aer_summary["measurement_rounds"]))
    aer_weighted_p = float(aer_summary["weighted_mean_p"])
    aer_simple_p = float(aer_summary["simple_mean_p"])
    aer_success = 1.0 - aer_weighted_p
    effective_success = float(stats.survival_prob)

    compare_row: Dict[str, object] = {
        "logical_state": args.logical_state,
        "seed": args.seed,
        "mode": args.mode,
        "n_rounds_aer": args.n_rounds,
        "max_rounds_effective": params.max_rounds,
        "k_requested": args.k,
        "k_effective": effective_k,
        "schedule_match": int(aer_meas_rounds == effective_meas_rounds),
        "sigma_z": params.sigma_z,
        "zeta": params.zeta,
        "tau_int": params.tau_int,
        "eta_t": params.eta_t,
        "t_m": params.t_m,
        "t_r": params.t_r,
        "p_m": params.p_m,
        "p_r": params.p_r,
        "gamma_x": params.gamma_x,
        "chi": params.chi,
        "aer_noise": int(args.aer_noise),
        "aer_metric": args.aer_metric,
        "aer_shots_total": args.shots_total,
        "aer_batch_M": args.batch_M,
        "aer_measurement_rounds": aer_meas_rounds,
        "aer_weighted_mean_p": aer_weighted_p,
        "aer_simple_mean_p": aer_simple_p,
        "aer_success_prob": aer_success,
        "effective_shots": stats.shots,
        "effective_measurement_rounds": effective_meas_rounds,
        "effective_logical_failure_prob": stats.logical_failure_prob,
        "effective_survival_prob": effective_success,
        "effective_lifetime_time": stats.lifetime_time,
        "effective_lifetime_rounds": stats.lifetime_rounds,
        "effective_measurement_efficiency": stats.measurement_efficiency,
        "effective_horizon_time": effective_horizon_time,
        "delta_success_effective_minus_aer": effective_success - aer_success,
    }

    compare_path = f"{out_prefix}_compare.csv"
    _write_compare_csv(compare_path, compare_row)

    effective_stats_path = f"{out_prefix}_effective_stats.csv"
    _write_compare_csv(effective_stats_path, asdict(stats))

    print(f"Saved: {aer_prefix}_summary.csv")
    print(f"Saved: {compare_path}")
    print(f"Saved: {effective_stats_path}")
    print(
        f"AER success={aer_success:.6f} | Effective survival={effective_success:.6f} "
        f"| schedule_match={compare_row['schedule_match']}"
    )


if __name__ == "__main__":
    main()
