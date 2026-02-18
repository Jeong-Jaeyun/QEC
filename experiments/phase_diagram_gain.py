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
    ap = argparse.ArgumentParser(description="Phase diagram for gain and measurement-normalized gain.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shots", "--trajectories", dest="shots", type=int, default=700)
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
    ap.add_argument("--eta_points", type=int, default=6)
    ap.add_argument("--pm_min", type=float, default=1e-3)
    ap.add_argument("--pm_max", type=float, default=1e-1)
    ap.add_argument("--pm_points", type=int, default=6)

    ap.add_argument("--p_m", "--pm", dest="p_m", type=float, default=None, help="Fixed p_m. If set, pm sweep is disabled.")
    ap.add_argument("--p_r", "--pr", dest="p_r", type=float, default=None, help="If set, fixes p_r independently of p_m.")
    ap.add_argument("--alpha", "--pr_ratio", dest="alpha", type=float, default=1.0, help="If --p_r is unset, p_r = alpha * p_m.")

    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=14)
    ap.add_argument("--k", type=int, default=None, help="Optional fixed k. If set, uses k_min=k_max=k.")

    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_png_gain", type=str, default=None)
    ap.add_argument("--out_png_mgain", type=str, default=None)
    ap.add_argument("--out_crossover_csv", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_csv:
        args.out_csv = build_result_path(
            "phase_diagram_gain",
            gamma_x=args.gamma_x,
            chi=args.chi,
            zeta=args.zeta,
            seed=args.seed,
            shots=args.shots,
            rounds=args.max_rounds,
        )
    if not args.out_png_gain:
        args.out_png_gain = args.out_csv.replace(".csv", "_Gstar.png")
    if not args.out_png_mgain:
        args.out_png_mgain = args.out_csv.replace(".csv", "_Gmstar.png")
    if not args.out_crossover_csv:
        args.out_crossover_csv = args.out_csv.replace(".csv", "_crossover.csv")
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
    x_gain: List[float] = []
    y_gain: List[float] = []
    z_gain: List[float] = []
    z_mgain: List[float] = []
    by_eta: dict[float, List[tuple[float, float, float]]] = {}
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

            crossover = int((auto_sweep.best_gain > 1.0) and (naive_sweep.best_gain <= 1.0))

            rows.append(
                make_result_row(
                    experiment="phase_diagram_gain",
                    mode=MODE_FULL,
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
                    k=1,
                    k_star=auto_sweep.k_star,
                    k_star_naive=naive_sweep.k_star,
                    lifetime_time=full.lifetime_time,
                    lifetime_time_median=full.lifetime_time_median,
                    lifetime_time_ci_low=full.lifetime_time_ci_low,
                    lifetime_time_ci_high=full.lifetime_time_ci_high,
                    lifetime_rounds=full.lifetime_rounds,
                    logical_failure_prob=full.logical_failure_prob,
                    survival_prob=full.survival_prob,
                    avg_measurements_to_fail=full.avg_measurements_to_fail,
                    measurement_efficiency=full.measurement_efficiency,
                    gain_vs_full=1.0,
                    measurement_gain_vs_full=1.0,
                    notes=f"crossover={crossover}",
                )
            )

            for pt in naive_sweep.points:
                rows.append(
                make_result_row(
                        experiment="phase_diagram_gain",
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
                        k=pt.k,
                        k_star=auto_sweep.k_star,
                        k_star_naive=naive_sweep.k_star,
                        lifetime_time=pt.lifetime_time,
                        logical_failure_prob=pt.logical_failure_prob,
                        measurement_efficiency=pt.measurement_efficiency,
                        gain_vs_full=pt.gain_vs_full,
                        measurement_gain_vs_full=pt.measurement_gain_vs_full,
                        notes=f"crossover={crossover}",
                    )
                )

            for pt in auto_sweep.points:
                rows.append(
                make_result_row(
                        experiment="phase_diagram_gain",
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
                        k=pt.k,
                        k_star=auto_sweep.k_star,
                        k_star_naive=naive_sweep.k_star,
                        lifetime_time=pt.lifetime_time,
                        logical_failure_prob=pt.logical_failure_prob,
                        measurement_efficiency=pt.measurement_efficiency,
                        gain_vs_full=pt.gain_vs_full,
                        measurement_gain_vs_full=pt.measurement_gain_vs_full,
                        notes=f"crossover={crossover}",
                    )
                )

            x_gain.append(eta_t)
            y_gain.append(p_m)
            z_gain.append(float(auto_sweep.best_gain))
            z_mgain.append(float(auto_sweep.best_measurement_gain))
            by_eta.setdefault(float(eta_t), []).append((float(p_m), float(auto_sweep.best_gain), float(naive_sweep.best_gain)))

            print(
                f"[{done}/{total}] eta_t={eta_t:.4g} p_m={p_m:.4g} p_r={p_r:.4g} "
                f"best_gain_auto={auto_sweep.best_gain:.4f} best_gain_naive={naive_sweep.best_gain:.4f} "
                f"k*_auto={auto_sweep.k_star}"
            )

    write_result_csv(args.out_csv, rows)
    print(f"Saved: {args.out_csv} ({len(rows)} rows)")

    ok_gain = save_scatter_heatmap(
        x=x_gain,
        y=y_gain,
        z=z_gain,
        out_png=args.out_png_gain,
        title="G* phase diagram (autonomous sparse)",
        xlabel="eta_t",
        ylabel="p_m",
        x_log=True,
        y_log=True,
        colorbar_label="G*",
    )
    if ok_gain:
        print(f"Saved: {args.out_png_gain}")
    else:
        print("Skipped PNG (matplotlib unavailable): G* heatmap")

    ok_mgain = save_scatter_heatmap(
        x=x_gain,
        y=y_gain,
        z=z_mgain,
        out_png=args.out_png_mgain,
        title="Gm* phase diagram (autonomous sparse)",
        xlabel="eta_t",
        ylabel="p_m",
        x_log=True,
        y_log=True,
        colorbar_label="Gm*",
    )
    if ok_mgain:
        print(f"Saved: {args.out_png_mgain}")
    else:
        print("Skipped PNG (matplotlib unavailable): Gm* heatmap")

    with open(args.out_crossover_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eta_t",
                "pm_best_crossover",
                "auto_gain_at_best",
                "naive_gain_at_best",
                "pm_nearest_auto_cross",
                "auto_gain_nearest_1",
            ]
        )
        for eta in sorted(by_eta.keys()):
            pts = sorted(by_eta[eta], key=lambda t: t[0])
            valid = [p for p in pts if (p[1] > 1.0 and p[2] <= 1.0)]
            if valid:
                pm_best, g_auto_best, g_naive_best = max(valid, key=lambda t: t[0])
            else:
                pm_best, g_auto_best, g_naive_best = "", "", ""
            pm_cross, g_cross, _ = min(pts, key=lambda t: abs(t[1] - 1.0))
            w.writerow([eta, pm_best, g_auto_best, g_naive_best, pm_cross, g_cross])
    print(f"Saved: {args.out_crossover_csv}")


if __name__ == "__main__":
    main()
