import argparse
import os
import csv
from qiskit_aer import AerSimulator
from core.circuit import CoreParams, build_core_circuit, build_core_circuit_with_syndrome
from core.noise import NoiseParams, build_noise_model
from scenarios.base import make_hook as hook_s0
from scenarios.nonlocal_buffer import make_hook as hook_s1
from scenarios.topo_minimal import make_hooks as hook_s2
from core.metrics import fidelity_data_2q, syndrome_stats

# hook은 S2와 동일(=syndrome logging 회로)
# 디코더는 utils.logging에서 선택
from utils.logging import (
    score_decoded_success,
    decode_flat_majority,
    decode_block_consensus,
)
from utils.plotting import plot_lines_logx

def _make_noise(base: NoiseParams, link_mult: float, inject_link: bool = True) -> NoiseParams:
    return NoiseParams(
        p1_data=base.p1_data,
        p1_anc=base.p1_anc,
        p2_data_data=base.p2_data_data,
        p2_data_anc=(min(0.45, base.p2_data_anc * link_mult) if inject_link else 0.0),
        pid_data=base.pid_data,
        pid_anc=base.pid_anc,
        ro_anc=base.ro_anc,
    )

def run_state_compare(args):
    """
    State-based comparison:
      - density_matrix backend
      - metric: fidelity (core.metrics 또는 기존 방식)
    """
    os.makedirs("results", exist_ok=True)

    # ----- params -----
    core_p = CoreParams(theta=args.theta, n_cycles=args.n_cycles,
                        idle_ticks_data=args.idle_data, idle_ticks_anc=args.idle_anc)
    core_p.save_density = True

    base_noise = NoiseParams(
        p1_data=args.p1_data, p1_anc=args.p1_anc,
        p2_data_data=args.p2_data_data, p2_data_anc=args.p2_data_anc,
        pid_data=args.pid_data, pid_anc=args.pid_anc,
        ro_anc=args.ro_anc
    )

    link_mults = args.link_mults

    # ----- scenario hooks -----
    # S0/S1/S2는 "hook"으로 모델링
    H0 = hook_s0(args)
    H1 = hook_s1(args)
    # S2는 topo_minimal에서 "constraint only"와 "constraint+feedback"을 분리해둘 수 있음
    # 여기서는 일단 S2a/S2b를 모두 비교하도록 만든다.
    H2a, H2b, Hlog = hook_s2(args)  # (constraint-only, constraint+feedback)

    # ----- simulator -----
    # state 기반에서는 density_matrix를 쓰는 게 명확
    ys = {"S0_BASE": [], "S1_NONLOCAL_BUFFER": [], "S2a_TOPO_MIN": [], "S2b_TOPO_MIN": []}

    for lm in link_mults:
        noise = _make_noise(base_noise, lm)
        nm = build_noise_model(noise)
        sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=args.seed)

        # build + run per scenario (각각 회로를 새로 만들어도 되고, hook만 다르게)
        # build_core_circuit(core_p, hook=..., ...) 형태로 통일하는 게 좋다.
        for label, hook in [("S0_BASE", H0), ("S1_NONLOCAL_BUFFER", H1), ("S2a_TOPO_MIN", H2a), ("S2b_TOPO_MIN", H2b)]:
            qc = build_core_circuit(core_p, hook=hook)
            result = sim.run(qc).result()
            # fidelity 계산은 너의 기존 방식(core.metrics or circuit helper)을 호출해야 함
            fid = fidelity_data_2q(result, qc, data_qubits=(0, 1), ideal="00")  
            ys[label].append(fid)

        print(f"[state] link_mult={lm} | " + " ".join([f"{k}={ys[k][-1]:.6f}" for k in ys]))

    # save csv
    out_csv = os.path.join("results", "compare_state.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult"] + list(ys.keys()))
        for i, lm in enumerate(link_mults):
            w.writerow([lm] + [ys[k][i] for k in ys])

    # plot
    out_png = os.path.join("results", "compare_state.png")
    plot_lines_logx(
        link_mults, ys,
        title="State Comparison: S0 vs S1 vs S2a vs S2b",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="state fidelity",
        out_png=out_png
    )
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


def run_decoded_compare(args):
    """
    Decoded comparison:
    - syndrome logging circuit
    - qasm(automatic) + shots
    - metric: decoded success probability
    """
    os.makedirs("results", exist_ok=True)

    core_p = CoreParams(theta=args.theta, n_cycles=args.n_cycles,
                        idle_ticks_data=args.idle_data, idle_ticks_anc=args.idle_anc)

    base_noise = NoiseParams(
        p1_data=args.p1_data, p1_anc=args.p1_anc,
        p2_data_data=args.p2_data_data, p2_data_anc=args.p2_data_anc,
        pid_data=args.pid_data, pid_anc=args.pid_anc,
        ro_anc=args.ro_anc
    )

    link_mults = args.link_mults

    # 회로 hook은 "로그 생성용"만 필요 (S2(A)와 동일한 회로를 사용)
    # topo_minimal.py에서 (reps=1, uncompute=False, feedback=False)로 hook을 리턴하도록 맞춰두는 게 깔끔
    _, _, Hlog = hook_s2(args)  # hook_s2가 (constraint-only, log-only) 형태로도 리턴하도록 구성 권장

    # 디코더 정책
    dec_flat = decode_flat_majority
    dec_block = (lambda s: decode_block_consensus(s, W=args.block_W))

    ys = {"S2_flat_majority": []}
    for W in args.W_list:
        ys[f"S3_block_consensus_W{W}"] = []

    ms = {"detection": [], "fn": [] , "fp": []}

    for lm in link_mults:
        # ---------- Injection ON ----------
        noise_on = _make_noise(base_noise, lm, inject_link=True)
        nm_on = build_noise_model(noise_on)
        sim_on = AerSimulator(method="automatic", noise_model=nm_on, seed_simulator=args.seed)

        qc = build_core_circuit_with_syndrome(core_p, hook=Hlog, n_cycles=core_p.n_cycles)
        result_on = sim_on.run(qc, shots=args.shots).result()
        counts_on = result_on.get_counts(qc)

        stats_on = syndrome_stats(counts_on, core_p.n_cycles)
        ms["detection"].append(stats_on["detection_rate"])
        ms["fn"].append(stats_on["false_negative"])

        # decoded scores (ON counts로만)
        s2 = score_decoded_success(counts_on, core_p.n_cycles, decode_flat_majority,
                           ideal_data=args.ideal, flip_target=args.flip_target)
        ys["S2_flat_majority"].append(s2)

        for W in args.W_list:
            dec_block = (lambda s, _W=W: decode_block_consensus(s, W=_W))  # W 캡처 중요
            s3 = score_decoded_success(counts_on, core_p.n_cycles, dec_block,
                                    ideal_data=args.ideal, flip_target=args.flip_target)
            ys[f"S3_block_consensus_W{W}"].append(s3)


        # ---------- Injection OFF (FP) ----------
        noise_off = _make_noise(base_noise, lm, inject_link=False)
        nm_off = build_noise_model(noise_off)
        sim_off = AerSimulator(method="automatic", noise_model=nm_off, seed_simulator=args.seed)

        # 회로는 동일하게 (hook 동일)
        qc_off = build_core_circuit_with_syndrome(core_p, hook=Hlog, n_cycles=core_p.n_cycles)
        result_off = sim_off.run(qc_off, shots=args.shots).result()
        counts_off = result_off.get_counts(qc_off)

        stats_off = syndrome_stats(counts_off, core_p.n_cycles)
        ms["fp"].append(stats_off["detection_rate"])  # OFF에서 P(s != 0) = FP
        s3_ref = ys[f"S3_block_consensus_W{args.block_W}"][-1]

        print(f"[decoded] link_mult={lm} | S2={s2:.6f} | S3(W={args.block_W})={s3_ref:.6f} | "f"det={ms['detection'][-1]:.4f} fn={ms['fn'][-1]:.4f} fp={ms['fp'][-1]:.4f}")

    W_scores = {}
    for W in args.W_list:
        key = f"S3_block_consensus_W{W}"
        vals = ys.get(key, [])
        if len(vals) != len(link_mults):
            raise RuntimeError(f"W={W} missing results: got {len(vals)} / {len(link_mults)}")
        W_scores[W] = sum(vals) / len(vals)

    out_csv = os.path.join("results", "compare_decoded.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult"] + list(ys.keys()) + ["detection", "fn", "fp"])
        for i, lm in enumerate(link_mults):
            w.writerow([lm] + [ys[k][i] for k in ys] + [ms["detection"][i], ms["fn"][i], ms["fp"][i]])
    
    out_csv2 = os.path.join("results", "fig4_W_score.csv")
    with open(out_csv2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["W", "mean_success"])
        for W in args.W_list:
            w.writerow([W, W_scores[W]])
    print(f"Saved: {out_csv2}")

    out_png = os.path.join("results", "compare_decoded.png")
    plot_lines_logx(
        link_mults, ys,
        title="Decoded Comparison: S2(A) vs S3(block consensus)",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="decoded success probability",
        out_png=out_png
    )
    out_png2 = os.path.join("results", "fig3_syndrome_stats.png")
    plot_lines_logx(
        link_mults, ms,
        title="Syndrome Observability (Fig.3)",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="rate",
        out_png=out_png2
    )
    out_png4 = os.path.join("results", "fig4_W_sweep.png")
    plot_lines_logx(
        link_mults, ys,
        title="Fig.4(a) Decoder Sensitivity: W sweep",
        xlabel="data-ancilla link noise multiplier (log scale)",
        ylabel="decoded success probability",
        out_png=out_png4
    )

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_png2}")
    print(f"Saved: {out_png4}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="scenario yaml (overlay). ex) config/topo.yaml")
    p.add_argument("--base_config", type=str, default=None, help="base yaml. default: config/base.yaml if exists")

    p.add_argument("--mode", choices=["state", "decoded"], required=True)

    # common
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--theta", type=float, default=0.20)
    p.add_argument("--n_cycles", type=int, default=20)
    p.add_argument("--idle_data", type=int, default=1)
    p.add_argument("--idle_anc", type=int, default=1)
    p.add_argument("--W_list", type=int, nargs="+", default=[1, 2, 4, 5, 10])


    # noise base
    p.add_argument("--p1_data", type=float, default=0.001)
    p.add_argument("--p1_anc", type=float, default=0.001)
    p.add_argument("--p2_data_data", type=float, default=0.002)
    p.add_argument("--p2_data_anc", type=float, default=0.005)
    p.add_argument("--pid_data", type=float, default=0.001)
    p.add_argument("--pid_anc", type=float, default=0.001)
    p.add_argument("--ro_anc", type=float, default=0.01)

    # decoded params
    p.add_argument("--shots", type=int, default=8000)
    p.add_argument("--ideal", type=str, default="00")
    p.add_argument("--flip_target", type=int, choices=[0, 1], default=0)
    p.add_argument("--block_W", type=int, default=4)

    # sweep points (고정 리스트)
    p.add_argument("--link_mults", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    
    args = p.parse_args()
    
    from utils.config import apply_cfg_to_args, load_config
    cfg = load_config(args.base_config, args.config)
    apply_cfg_to_args(args, cfg)
    return args


def main():
    args = parse_args()
    if args.mode == "state":
        run_state_compare(args)
    else:
        run_decoded_compare(args)


if __name__ == "__main__":
    main()
