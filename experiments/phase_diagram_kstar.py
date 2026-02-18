from __future__ import annotations

import argparse
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
    MODE_NAIVE,
    EffectiveModelParams,
    lifetime_gain,
    measurement_normalized_gain,
    simulate_lifetime,
    sweep_k,
)
from experiments._plotting import save_scatter_heatmap
from experiments._result_schema import build_result_path, logspace, make_result_row, write_result_csv


def _resolve_pr(pm: float, p_r: Optional[float], alpha: float) -> float:
    if p_r is not None:
        return max(0.0, min(0.49, float(p_r)))
    return max(0.0, min(0.49, float(alpha) * float(pm)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase diagram for k* (optimal sparse monitoring interval).")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shots", "--trajectories", dest="shots", type=int, default=800)
    ap.add_argument("--max_rounds", "--n_rounds_max", dest="max_rounds", type=int, default=300)

    ap.add_argument("--gamma_x", type=float, default=0.03)
    ap.add_argument("--chi", type=float, default=4.0)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--sigma_z", type=float, default=None, help="Optional drift std. If set, zeta=sigma_z*tau_int.")
    ap.add_argument("--tau_int", type=float, default=1.0)
    ap.add_argument("--eta_t", type=float, default=None, help="Optional fixed eta_t. Disables eta sweep.")
    ap.add_argument("--t_m", "--tm", dest="t_m", type=float, default=None, help="Optional fixed measurement latency.")
    ap.add_argument("--t_r", "--tr", dest="t_r", type=float, default=None, help="Optional fixed reset latency.")

    ap.add_argument("--eta_min", type=float, default=0.01)
    ap.add_argument("--eta_max", type=float, default=3.0)
    ap.add_argument("--eta_points", type=int, default=8)
    ap.add_argument("--pm_min", type=float, default=1e-3)
    ap.add_argument("--pm_max", type=float, default=1e-1)
    ap.add_argument("--pm_points", type=int, default=7)

    ap.add_argument("--p_m", "--pm", dest="p_m", type=float, default=None, help="Fixed p_m. If set, pm sweep is disabled.")
    ap.add_argument("--p_r", "--pr", dest="p_r", type=float, default=None, help="If set, fixes p_r independently of p_m.")
    ap.add_argument("--alpha", "--pr_ratio", dest="alpha", type=float, default=1.0, help="If --p_r is unset, p_r = alpha * p_m.")

    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--k", type=int, default=None, help="Optional fixed k. If set, uses k_min=k_max=k.")

    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_png", type=str, default=None)
    ap.add_argument("--out_png_naive", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_csv:
        args.out_csv = build_result_path(
            "phase_diagram_kstar",
            gamma_x=args.gamma_x,
            chi=args.chi,
            zeta=args.zeta,
            seed=args.seed,
            shots=args.shots,
            rounds=args.max_rounds,
        )
    if not args.out_png:
        args.out_png = args.out_csv.replace(".csv", "_auto.png")
    if not args.out_png_naive:
        args.out_png_naive = args.out_csv.replace(".csv", "_naive.png")
    if args.eta_t is not None:
        eta_values = [float(args.eta_t)]
    elif args.t_m is not None or args.t_r is not None:
        eta_values = [
            (float(args.t_m if args.t_m is not None else 0.0) + float(args.t_r if args.t_r is not None else 0.0))
            / float(args.tau_int)
        ]
    else:
        eta_values = logspace(args.eta_min, args.eta_max, args.eta_points)
    pm_values = [float(args.p_m)] if args.p_m is not None else logspace(args.pm_min, args.pm_max, args.pm_points)
    zeta_value = (float(args.sigma_z) * float(args.tau_int)) if args.sigma_z is not None else float(args.zeta)
    sigma_value = float(args.sigma_z) if args.sigma_z is not None else (zeta_value / float(args.tau_int))
    if args.k is not None:
        k_values = [max(2, int(args.k))]
    else:
        k_values = list(range(max(2, int(args.k_min)), int(args.k_max) + 1))
    if not k_values:
        raise ValueError("k range is empty. Check --k_min/--k_max.")

    rows: List[dict] = []
    x_auto: List[float] = []
    y_auto: List[float] = []
    z_auto: List[float] = []
    x_naive: List[float] = []
    y_naive: List[float] = []
    z_naive: List[float] = []
    total = len(eta_values) * len(pm_values)
    done = 0
    for eta_t in eta_values:
        for p_m in pm_values:
            done += 1
            p_r = _resolve_pr(p_m, args.p_r, args.alpha)
            base = EffectiveModelParams(
                gamma_x=args.gamma_x,
                chi=args.chi,
                zeta=zeta_value,
                sigma_z=sigma_value,
                eta_t=eta_t,
                t_m=args.t_m,
                t_r=args.t_r,
                tau_int=args.tau_int,
                p_m=p_m,
                p_r=p_r,
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
                    experiment="phase_diagram_kstar",
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
                    alpha_pr_over_pm=(args.alpha if args.p_r is None else ""),
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
            x_auto.append(eta_t)
            y_auto.append(p_m)
            z_auto.append(float(auto_sweep.k_star))
            rows.append(
                make_result_row(
                    experiment="phase_diagram_kstar",
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
                    alpha_pr_over_pm=(args.alpha if args.p_r is None else ""),
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
            x_naive.append(eta_t)
            y_naive.append(p_m)
            z_naive.append(float(naive_sweep.k_star))

            print(
                f"[{done}/{total}] eta_t={eta_t:.4g} p_m={p_m:.4g} p_r={p_r:.4g} "
                f"k*_auto={auto_sweep.k_star} k*_naive={naive_sweep.k_star}"
            )

    write_result_csv(args.out_csv, rows)
    print(f"Saved: {args.out_csv} ({len(rows)} rows)")

    ok_auto = save_scatter_heatmap(
        x=x_auto,
        y=y_auto,
        z=z_auto,
        out_png=args.out_png,
        title="k* phase diagram (autonomous sparse)",
        xlabel="eta_t",
        ylabel="p_m",
        x_log=True,
        y_log=True,
        colorbar_label="k*",
    )
    if ok_auto:
        print(f"Saved: {args.out_png}")
    else:
        print("Skipped PNG (matplotlib unavailable): autonomous k* heatmap")

    ok_naive = save_scatter_heatmap(
        x=x_naive,
        y=y_naive,
        z=z_naive,
        out_png=args.out_png_naive,
        title="k* phase diagram (naive sparse)",
        xlabel="eta_t",
        ylabel="p_m",
        x_log=True,
        y_log=True,
        colorbar_label="k*",
    )
    if ok_naive:
        print(f"Saved: {args.out_png_naive}")
    else:
        print("Skipped PNG (matplotlib unavailable): naive k* heatmap")


if __name__ == "__main__":
    main()
