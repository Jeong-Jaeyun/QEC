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
    simulate_logical_coherence,
)
from experiments._plotting import save_line_plot
from experiments._result_schema import build_result_path, make_result_row, write_result_csv


def _resolve_pr(pm: float, p_r: Optional[float], alpha: float) -> float:
    if p_r is not None:
        return max(0.0, min(0.49, float(p_r)))
    return max(0.0, min(0.49, float(alpha) * float(pm)))


def _resolve_eta(eta_t: float, t_m: Optional[float], t_r: Optional[float], tau_int: float) -> float:
    if t_m is None and t_r is None:
        return float(eta_t)
    return (float(t_m if t_m is not None else 0.0) + float(t_r if t_r is not None else 0.0)) / float(tau_int)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Logical coherence track C_L(t)=<X1X2X3>.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shots", "--trajectories", dest="shots", type=int, default=1600, help="Shots for coherence simulation.")
    ap.add_argument(
        "--shots_lifetime",
        "--trajectories_lifetime",
        dest="shots_lifetime",
        type=int,
        default=1000,
        help="Shots for lifetime references.",
    )
    ap.add_argument("--max_rounds", "--n_rounds_max", dest="max_rounds", type=int, default=320)
    ap.add_argument("--sample_rounds", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 96])

    ap.add_argument("--gamma_x", type=float, default=0.03)
    ap.add_argument("--chi", type=float, default=4.0)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--sigma_z", type=float, default=None, help="Optional drift std. If set, zeta=sigma_z*tau_int.")
    ap.add_argument("--eta_t", type=float, default=1.0)
    ap.add_argument("--t_m", "--tm", dest="t_m", type=float, default=None)
    ap.add_argument("--t_r", "--tr", dest="t_r", type=float, default=None)
    ap.add_argument("--tau_int", type=float, default=1.0)
    ap.add_argument("--p_m", "--pm", dest="p_m", type=float, default=0.02)
    ap.add_argument("--p_r", "--pr", dest="p_r", type=float, default=None)
    ap.add_argument("--alpha", "--pr_ratio", dest="alpha", type=float, default=1.0)
    ap.add_argument("--k_sparse", "--k", dest="k_sparse", type=int, default=6)

    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_png", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_csv:
        args.out_csv = build_result_path(
            "coherence_experiment",
            gamma_x=args.gamma_x,
            chi=args.chi,
            zeta=args.zeta,
            eta_t=args.eta_t,
            pm=args.p_m,
            alpha=args.alpha if args.p_r is None else "",
            seed=args.seed,
            shots=args.shots,
        )
    if not args.out_png:
        args.out_png = args.out_csv.replace(".csv", ".png")
    if args.k_sparse <= 1:
        raise ValueError("--k_sparse must be > 1 for sparse baselines.")

    p_r = _resolve_pr(args.p_m, args.p_r, args.alpha)
    eta_eff = _resolve_eta(args.eta_t, args.t_m, args.t_r, args.tau_int)
    zeta_eff = (float(args.sigma_z) * float(args.tau_int)) if args.sigma_z is not None else float(args.zeta)
    sigma_eff = float(args.sigma_z) if args.sigma_z is not None else (zeta_eff / float(args.tau_int))
    base = EffectiveModelParams(
        gamma_x=args.gamma_x,
        chi=args.chi,
        zeta=zeta_eff,
        sigma_z=sigma_eff,
        eta_t=eta_eff,
        t_m=args.t_m,
        t_r=args.t_r,
        tau_int=args.tau_int,
        p_m=args.p_m,
        p_r=p_r,
        k=args.k_sparse,
        max_rounds=args.max_rounds,
        shots=args.shots_lifetime,
        seed=args.seed,
    )

    full_life = simulate_lifetime(replace(base, k=1), MODE_FULL)
    naive_life = simulate_lifetime(replace(base, k=args.k_sparse), MODE_NAIVE)
    auto_life = simulate_lifetime(replace(base, k=args.k_sparse), MODE_AUTO)

    full_coh = simulate_logical_coherence(
        replace(base, k=1),
        MODE_FULL,
        args.sample_rounds,
        shots=args.shots,
        seed=args.seed + 500,
    )
    naive_coh = simulate_logical_coherence(
        replace(base, k=args.k_sparse),
        MODE_NAIVE,
        args.sample_rounds,
        shots=args.shots,
        seed=args.seed + 600,
    )
    auto_coh = simulate_logical_coherence(
        replace(base, k=args.k_sparse),
        MODE_AUTO,
        args.sample_rounds,
        shots=args.shots,
        seed=args.seed + 700,
    )

    rows: List[dict] = []
    series = [
        (MODE_FULL, full_life, full_coh),
        (MODE_NAIVE, naive_life, naive_coh),
        (MODE_AUTO, auto_life, auto_coh),
    ]

    for mode, life, coh in series:
        g_ratio = coh.gamma_lphi / max(full_coh.gamma_lphi, 1e-12)
        for pt in coh.points:
            rows.append(
                make_result_row(
                    experiment="coherence_experiment",
                    mode=mode,
                    seed=args.seed,
                    shots=args.shots,
                    max_rounds=args.max_rounds,
                    gamma_x=args.gamma_x,
                    chi=args.chi,
                    sigma_z=base.sigma_z,
                    zeta=base.zeta,
                    eta_t=base.eta_t,
                    t_m=base.t_m,
                    t_r=base.t_r,
                    tau_int=args.tau_int,
                    p_m=args.p_m,
                    p_r=p_r,
                    alpha_pr_over_pm=(args.alpha if args.p_r is None else ""),
                    k=(1 if mode == MODE_FULL else args.k_sparse),
                    lifetime_time=life.lifetime_time,
                    lifetime_time_median=life.lifetime_time_median,
                    lifetime_time_ci_low=life.lifetime_time_ci_low,
                    lifetime_time_ci_high=life.lifetime_time_ci_high,
                    lifetime_rounds=life.lifetime_rounds,
                    logical_failure_prob=life.logical_failure_prob,
                    survival_prob=life.survival_prob,
                    avg_measurements_to_fail=life.avg_measurements_to_fail,
                    measurement_efficiency=life.measurement_efficiency,
                    gain_vs_full=lifetime_gain(life, full_life),
                    measurement_gain_vs_full=measurement_normalized_gain(life, full_life),
                    gamma_lphi=coh.gamma_lphi,
                    gamma_lphi_ci_low=coh.gamma_lphi_ci_low,
                    gamma_lphi_ci_high=coh.gamma_lphi_ci_high,
                    gamma_lphi_ratio_vs_full=g_ratio,
                    coherence_time=pt.time,
                    coherence_value=pt.coherence,
                )
            )

    write_result_csv(args.out_csv, rows)
    print(f"Saved: {args.out_csv} ({len(rows)} rows)")
    print(f"Gamma_Lphi(full)={full_coh.gamma_lphi:.6g}")
    print(f"Gamma_Lphi(naive)={naive_coh.gamma_lphi:.6g}")
    print(f"Gamma_Lphi(auto)={auto_coh.gamma_lphi:.6g}")

    x = [pt.time for pt in full_coh.points]
    series = [
        ("full", [pt.coherence for pt in full_coh.points]),
        ("naive", [pt.coherence for pt in naive_coh.points]),
        ("auto", [pt.coherence for pt in auto_coh.points]),
    ]
    ok = save_line_plot(
        x=x,
        series=series,
        out_png=args.out_png,
        title="Logical coherence C_L(t)=<X1X2X3>",
        xlabel="time",
        ylabel="C_L(t)",
        x_log=False,
    )
    if ok:
        print(f"Saved: {args.out_png}")
    else:
        print("Skipped PNG (matplotlib unavailable): coherence curve")


if __name__ == "__main__":
    main()
