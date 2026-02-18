from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.effective_model import (
    MODE_AUTO,
    MODE_FULL,
    EffectiveModelParams,
    is_safe_zone,
    lifetime_gain,
    measurement_normalized_gain,
    simulate_lifetime,
    simulate_logical_coherence,
)
from experiments._plotting import save_scatter_heatmap
from experiments._result_schema import build_result_path, logspace, make_result_row, write_result_csv


def _resolve_pr(pm: float, p_r: Optional[float], alpha: float) -> float:
    if p_r is not None:
        return max(0.0, min(0.49, float(p_r)))
    return max(0.0, min(0.49, float(alpha) * float(pm)))


def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(num - 1)
    return [float(start) + i * step for i in range(num)]


def _resolve_eta(eta_t: float, t_m: Optional[float], t_r: Optional[float], tau_int: float) -> float:
    if t_m is None and t_r is None:
        return float(eta_t)
    return (float(t_m if t_m is not None else 0.0) + float(t_r if t_r is not None else 0.0)) / float(tau_int)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Safe-zone sweep over (chi, zeta).")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shots", "--trajectories", dest="shots", type=int, default=900, help="Shots for lifetime simulation.")
    ap.add_argument(
        "--shots_coherence",
        "--trajectories_coherence",
        dest="shots_coherence",
        type=int,
        default=700,
        help="Shots for coherence simulation.",
    )
    ap.add_argument("--max_rounds", "--n_rounds_max", dest="max_rounds", type=int, default=320)
    ap.add_argument("--sample_rounds", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])

    ap.add_argument("--gamma_x", type=float, default=0.03)
    ap.add_argument("--tau_int", type=float, default=1.0)
    ap.add_argument("--eta_t", type=float, default=1.0)
    ap.add_argument("--t_m", "--tm", dest="t_m", type=float, default=None, help="Optional measurement latency.")
    ap.add_argument("--t_r", "--tr", dest="t_r", type=float, default=None, help="Optional reset latency.")
    ap.add_argument("--p_m", "--pm", dest="p_m", type=float, default=0.02)
    ap.add_argument("--p_r", "--pr", dest="p_r", type=float, default=None)
    ap.add_argument("--alpha", "--pr_ratio", dest="alpha", type=float, default=1.0)

    ap.add_argument("--k_sparse", "--k", dest="k_sparse", type=int, default=6)
    ap.add_argument("--chi_min", type=float, default=0.0)
    ap.add_argument("--chi_max", type=float, default=10.0)
    ap.add_argument("--chi_points", type=int, default=11)

    ap.add_argument("--zeta_ratio_min", type=float, default=0.01)
    ap.add_argument("--zeta_ratio_max", type=float, default=0.5)
    ap.add_argument("--zeta_points", type=int, default=9)

    ap.add_argument("--epsilon_tau", type=float, default=0.1)
    ap.add_argument("--epsilon_phi", type=float, default=0.1)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_png_gain", type=str, default=None)
    ap.add_argument("--out_png_mask", type=str, default=None)
    ap.add_argument("--out_summary_csv", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_csv:
        args.out_csv = build_result_path(
            "safe_zone",
            gamma_x=args.gamma_x,
            eta_t=args.eta_t,
            pm=args.p_m,
            alpha=args.alpha if args.p_r is None else "",
            seed=args.seed,
            shots=args.shots,
        )
    if not args.out_png_gain:
        args.out_png_gain = args.out_csv.replace(".csv", "_gain.png")
    if not args.out_png_mask:
        args.out_png_mask = args.out_csv.replace(".csv", "_mask.png")
    if not args.out_summary_csv:
        args.out_summary_csv = args.out_csv.replace(".csv", "_summary.csv")
    if args.k_sparse <= 1:
        raise ValueError("--k_sparse must be > 1 for autonomous sparse mode.")

    zeta_values = [args.gamma_x * r for r in logspace(args.zeta_ratio_min, args.zeta_ratio_max, args.zeta_points)]
    chi_values = _linspace(args.chi_min, args.chi_max, args.chi_points)
    p_r = _resolve_pr(args.p_m, args.p_r, args.alpha)
    eta_eff = _resolve_eta(args.eta_t, args.t_m, args.t_r, args.tau_int)

    rows: List[dict] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    z_gain: List[float] = []
    z_mask: List[float] = []
    boundary_points: List[tuple[float, float, int]] = []
    total = len(zeta_values) * len(chi_values)
    done = 0

    for zeta in zeta_values:
        base = EffectiveModelParams(
            gamma_x=args.gamma_x,
            chi=0.0,
            sigma_z=zeta / max(args.tau_int, 1e-12),
            zeta=zeta,
            eta_t=eta_eff,
            t_m=args.t_m,
            t_r=args.t_r,
            tau_int=args.tau_int,
            p_m=args.p_m,
            p_r=p_r,
            k=args.k_sparse,
            max_rounds=args.max_rounds,
            shots=args.shots,
            seed=args.seed,
        )

        full_stats = simulate_lifetime(replace(base, k=1), MODE_FULL)
        full_coh = simulate_logical_coherence(
            replace(base, k=1),
            MODE_FULL,
            args.sample_rounds,
            shots=args.shots_coherence,
            seed=args.seed + 1000,
        )
        for chi in chi_values:
            done += 1
            auto_params = replace(base, chi=chi, k=args.k_sparse)
            auto_stats = simulate_lifetime(auto_params, MODE_AUTO)
            auto_coh = simulate_logical_coherence(
                auto_params,
                MODE_AUTO,
                args.sample_rounds,
                shots=args.shots_coherence,
                seed=args.seed + 2000,
            )

            lifetime_ratio = lifetime_gain(auto_stats, full_stats)
            gamma_ratio = auto_coh.gamma_lphi / max(full_coh.gamma_lphi, 1e-12)
            safe = is_safe_zone(
                lifetime_ratio,
                gamma_ratio,
                epsilon_tau=args.epsilon_tau,
                epsilon_phi=args.epsilon_phi,
            )

            rows.append(
                make_result_row(
                    experiment="safe_zone",
                    mode=MODE_AUTO,
                    seed=args.seed,
                    shots=args.shots,
                    max_rounds=args.max_rounds,
                    gamma_x=args.gamma_x,
                    chi=chi,
                    sigma_z=auto_params.sigma_z,
                    zeta=zeta,
                    eta_t=auto_params.eta_t,
                    t_m=auto_params.t_m,
                    t_r=auto_params.t_r,
                    tau_int=args.tau_int,
                    p_m=args.p_m,
                    p_r=p_r,
                    alpha_pr_over_pm=(args.alpha if args.p_r is None else ""),
                    k=args.k_sparse,
                    lifetime_time=auto_stats.lifetime_time,
                    lifetime_time_median=auto_stats.lifetime_time_median,
                    lifetime_time_ci_low=auto_stats.lifetime_time_ci_low,
                    lifetime_time_ci_high=auto_stats.lifetime_time_ci_high,
                    lifetime_rounds=auto_stats.lifetime_rounds,
                    logical_failure_prob=auto_stats.logical_failure_prob,
                    survival_prob=auto_stats.survival_prob,
                    avg_measurements_to_fail=auto_stats.avg_measurements_to_fail,
                    measurement_efficiency=auto_stats.measurement_efficiency,
                    gain_vs_full=lifetime_ratio,
                    measurement_gain_vs_full=measurement_normalized_gain(auto_stats, full_stats),
                    gamma_lphi=auto_coh.gamma_lphi,
                    gamma_lphi_ci_low=auto_coh.gamma_lphi_ci_low,
                    gamma_lphi_ci_high=auto_coh.gamma_lphi_ci_high,
                    gamma_lphi_ratio_vs_full=gamma_ratio,
                    safe_zone=int(safe),
                    notes=f"gamma_full={full_coh.gamma_lphi:.6g}",
                )
            )
            x_vals.append(zeta)
            y_vals.append(chi)
            z_gain.append(lifetime_ratio)
            safe_i = int(safe)
            z_mask.append(float(safe_i))
            boundary_points.append((float(zeta), float(chi), safe_i))

            print(
                f"[{done}/{total}] zeta={zeta:.4g} chi={chi:.4g} "
                f"life_ratio={lifetime_ratio:.4f} gamma_ratio={gamma_ratio:.4f} safe={int(safe)}"
            )

    write_result_csv(args.out_csv, rows)
    print(f"Saved: {args.out_csv} ({len(rows)} rows)")

    ok_gain = save_scatter_heatmap(
        x=x_vals,
        y=y_vals,
        z=z_gain,
        out_png=args.out_png_gain,
        title="Safe-zone scan: lifetime gain",
        xlabel="zeta",
        ylabel="chi",
        x_log=True,
        y_log=False,
        colorbar_label="lifetime gain",
    )
    if ok_gain:
        print(f"Saved: {args.out_png_gain}")
    else:
        print("Skipped PNG (matplotlib unavailable): safe-zone gain")

    ok_mask = save_scatter_heatmap(
        x=x_vals,
        y=y_vals,
        z=z_mask,
        out_png=args.out_png_mask,
        title="Safe-zone mask",
        xlabel="zeta",
        ylabel="chi",
        x_log=True,
        y_log=False,
        colorbar_label="safe (1=yes)",
        cmap="plasma",
    )
    if ok_mask:
        print(f"Saved: {args.out_png_mask}")
    else:
        print("Skipped PNG (matplotlib unavailable): safe-zone mask")

    safe_count = int(sum(int(v) for v in z_mask))
    total_points = len(z_mask)
    safe_fraction = float(safe_count) / float(total_points) if total_points > 0 else 0.0
    with open(args.out_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "safe_count", "total_points", "safe_fraction", "zeta", "chi_boundary_min"])
        w.writerow(["overview", safe_count, total_points, safe_fraction, "", ""])
        for zeta in sorted(set(v[0] for v in boundary_points)):
            chis = [chi for zz, chi, safe_i in boundary_points if zz == zeta and safe_i == 1]
            chi_boundary = min(chis) if chis else ""
            w.writerow(["boundary", "", "", "", zeta, chi_boundary])
    print(f"Saved: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
