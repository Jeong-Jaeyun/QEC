from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.effective_model import (
    MODE_AUTO,
    MODE_FULL,
    MODE_NAIVE,
    EffectiveModelParams,
    lifetime_gain,
    measurement_normalized_gain,
    simulate_lifetime,
    sweep_k,
)
from experiments._plotting import save_scatter_heatmap
from experiments._result_schema import build_result_path, logspace, make_result_row, write_result_csv


def _resolve_eta(eta_t: float, t_m: float | None, t_r: float | None, tau_int: float) -> float:
    if t_m is None and t_r is None:
        return float(eta_t)
    return (float(t_m if t_m is not None else 0.0) + float(t_r if t_r is not None else 0.0)) / float(tau_int)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase diagram over independent (p_m, p_r) grid.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shots", "--trajectories", dest="shots", type=int, default=700)
    ap.add_argument("--max_rounds", "--n_rounds_max", dest="max_rounds", type=int, default=300)

    ap.add_argument("--gamma_x", type=float, default=0.03)
    ap.add_argument("--chi", type=float, default=4.0)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--sigma_z", type=float, default=None, help="Optional drift std. If set, zeta=sigma_z*tau_int.")
    ap.add_argument("--tau_int", type=float, default=1.0)
    ap.add_argument("--eta_t", type=float, default=1.0)
    ap.add_argument("--t_m", "--tm", dest="t_m", type=float, default=None)
    ap.add_argument("--t_r", "--tr", dest="t_r", type=float, default=None)

    ap.add_argument("--pm_min", type=float, default=1e-3)
    ap.add_argument("--pm_max", type=float, default=1e-1)
    ap.add_argument("--pm_points", type=int, default=7)
    ap.add_argument("--pr_min", type=float, default=1e-3)
    ap.add_argument("--pr_max", type=float, default=1e-1)
    ap.add_argument("--pr_points", type=int, default=7)

    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=14)
    ap.add_argument("--k", type=int, default=None, help="Optional fixed k. If set, uses k_min=k_max=k.")

    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_png_gain", type=str, default=None)
    ap.add_argument("--out_png_mgain", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_csv:
        args.out_csv = build_result_path(
            "phase_diagram_pmpr",
            gamma_x=args.gamma_x,
            chi=args.chi,
            zeta=args.zeta,
            eta_t=args.eta_t,
            seed=args.seed,
            shots=args.shots,
            rounds=args.max_rounds,
        )
    if not args.out_png_gain:
        args.out_png_gain = args.out_csv.replace(".csv", "_Gstar.png")
    if not args.out_png_mgain:
        args.out_png_mgain = args.out_csv.replace(".csv", "_Gmstar.png")

    pm_values = logspace(args.pm_min, args.pm_max, args.pm_points)
    pr_values = logspace(args.pr_min, args.pr_max, args.pr_points)
    zeta_value = (float(args.sigma_z) * float(args.tau_int)) if args.sigma_z is not None else float(args.zeta)
    sigma_value = float(args.sigma_z) if args.sigma_z is not None else (zeta_value / float(args.tau_int))
    eta_eff = _resolve_eta(args.eta_t, args.t_m, args.t_r, args.tau_int)
    if args.k is not None:
        k_values = [max(2, int(args.k))]
    else:
        k_values = list(range(max(2, int(args.k_min)), int(args.k_max) + 1))
    if not k_values:
        raise ValueError("k range is empty. Check --k_min/--k_max.")

    rows: List[dict] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    z_gain: List[float] = []
    z_mgain: List[float] = []
    total = len(pm_values) * len(pr_values)
    done = 0

    for p_r in pr_values:
        for p_m in pm_values:
            done += 1
            base = EffectiveModelParams(
                gamma_x=args.gamma_x,
                chi=args.chi,
                zeta=zeta_value,
                sigma_z=sigma_value,
                eta_t=eta_eff,
                t_m=args.t_m,
                t_r=args.t_r,
                tau_int=args.tau_int,
                p_m=float(p_m),
                p_r=float(p_r),
                k=max(2, args.k_min),
                max_rounds=args.max_rounds,
                shots=args.shots,
                seed=args.seed,
            )

            full = simulate_lifetime(replace(base, k=1), MODE_FULL)
            naive_sweep = sweep_k(base, MODE_NAIVE, k_values, full_stats=full)
            auto_sweep = sweep_k(base, MODE_AUTO, k_values, full_stats=full)

            rows.append(
                make_result_row(
                    experiment="phase_diagram_pmpr",
                    mode=MODE_AUTO,
                    seed=args.seed,
                    shots=args.shots,
                    max_rounds=args.max_rounds,
                    gamma_x=base.gamma_x,
                    chi=base.chi,
                    sigma_z=base.sigma_z,
                    zeta=base.zeta,
                    eta_t=base.eta_t,
                    t_m=base.t_m,
                    t_r=base.t_r,
                    tau_int=base.tau_int,
                    p_m=base.p_m,
                    p_r=base.p_r,
                    k=auto_sweep.k_star,
                    k_star=auto_sweep.k_star,
                    k_star_naive=naive_sweep.k_star,
                    lifetime_time=auto_sweep.best_stats.lifetime_time,
                    lifetime_time_median=auto_sweep.best_stats.lifetime_time_median,
                    lifetime_time_ci_low=auto_sweep.best_stats.lifetime_time_ci_low,
                    lifetime_time_ci_high=auto_sweep.best_stats.lifetime_time_ci_high,
                    lifetime_rounds=auto_sweep.best_stats.lifetime_rounds,
                    logical_failure_prob=auto_sweep.best_stats.logical_failure_prob,
                    survival_prob=auto_sweep.best_stats.survival_prob,
                    avg_measurements_to_fail=auto_sweep.best_stats.avg_measurements_to_fail,
                    measurement_efficiency=auto_sweep.best_stats.measurement_efficiency,
                    gain_vs_full=lifetime_gain(auto_sweep.best_stats, full),
                    measurement_gain_vs_full=measurement_normalized_gain(auto_sweep.best_stats, full),
                    notes=f"naive_best_gain={naive_sweep.best_gain:.6f}",
                )
            )
            rows.append(
                make_result_row(
                    experiment="phase_diagram_pmpr",
                    mode=MODE_NAIVE,
                    seed=args.seed,
                    shots=args.shots,
                    max_rounds=args.max_rounds,
                    gamma_x=base.gamma_x,
                    chi=base.chi,
                    sigma_z=base.sigma_z,
                    zeta=base.zeta,
                    eta_t=base.eta_t,
                    t_m=base.t_m,
                    t_r=base.t_r,
                    tau_int=base.tau_int,
                    p_m=base.p_m,
                    p_r=base.p_r,
                    k=naive_sweep.k_star,
                    k_star=auto_sweep.k_star,
                    k_star_naive=naive_sweep.k_star,
                    lifetime_time=naive_sweep.best_stats.lifetime_time,
                    lifetime_time_median=naive_sweep.best_stats.lifetime_time_median,
                    lifetime_time_ci_low=naive_sweep.best_stats.lifetime_time_ci_low,
                    lifetime_time_ci_high=naive_sweep.best_stats.lifetime_time_ci_high,
                    lifetime_rounds=naive_sweep.best_stats.lifetime_rounds,
                    logical_failure_prob=naive_sweep.best_stats.logical_failure_prob,
                    survival_prob=naive_sweep.best_stats.survival_prob,
                    avg_measurements_to_fail=naive_sweep.best_stats.avg_measurements_to_fail,
                    measurement_efficiency=naive_sweep.best_stats.measurement_efficiency,
                    gain_vs_full=lifetime_gain(naive_sweep.best_stats, full),
                    measurement_gain_vs_full=measurement_normalized_gain(naive_sweep.best_stats, full),
                    notes=f"auto_best_gain={auto_sweep.best_gain:.6f}",
                )
            )

            x_vals.append(float(p_m))
            y_vals.append(float(p_r))
            z_gain.append(float(auto_sweep.best_gain))
            z_mgain.append(float(auto_sweep.best_measurement_gain))

            print(
                f"[{done}/{total}] p_m={p_m:.4g} p_r={p_r:.4g} "
                f"G*={auto_sweep.best_gain:.4f} Gm*={auto_sweep.best_measurement_gain:.4f} "
                f"k*_auto={auto_sweep.k_star}"
            )

    write_result_csv(args.out_csv, rows)
    print(f"Saved: {args.out_csv} ({len(rows)} rows)")

    ok_gain = save_scatter_heatmap(
        x=x_vals,
        y=y_vals,
        z=z_gain,
        out_png=args.out_png_gain,
        title="G* over (p_m, p_r)",
        xlabel="p_m",
        ylabel="p_r",
        x_log=True,
        y_log=True,
        colorbar_label="G*",
    )
    if ok_gain:
        print(f"Saved: {args.out_png_gain}")
    else:
        print("Skipped PNG (matplotlib unavailable): G* (pm-pr)")

    ok_mgain = save_scatter_heatmap(
        x=x_vals,
        y=y_vals,
        z=z_mgain,
        out_png=args.out_png_mgain,
        title="Gm* over (p_m, p_r)",
        xlabel="p_m",
        ylabel="p_r",
        x_log=True,
        y_log=True,
        colorbar_label="Gm*",
    )
    if ok_mgain:
        print(f"Saved: {args.out_png_mgain}")
    else:
        print("Skipped PNG (matplotlib unavailable): Gm* (pm-pr)")


if __name__ == "__main__":
    main()
