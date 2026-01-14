import matplotlib
matplotlib.use("Agg")

import os, csv
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

from core.circuit import CoreParams
from core.noise import NoiseParams, build_noise_model
from core.circuit import build_core_circuit_with_syndrome
from scenarios.topo_minimal import TopoMinimalParams, make_topo_hook


def majority(bits: str) -> int:
    ones = bits.count("1")
    zeros = len(bits) - ones
    return 1 if ones > zeros else 0


def decode_flat_majority(s_bits: str) -> int:
    """
    S2(A) baseline decoder: 
    - majority over all syndrome bits
    """
    return majority(s_bits)


def decode_block_consensus(s_bits: str, W: int) -> int:
    """
    S3 decoder:
    - split syndrome sequence into blocks of length W
    - block decision = majority(block)
    - consensus = majority(block decisions)
    """
    if W <= 0:
        return decode_flat_majority(s_bits)

    blocks = [s_bits[i:i+W] for i in range(0, len(s_bits), W)]
    # drop last short block optionally (or keep). MVP: keep.
    block_votes = [majority(b) for b in blocks if len(b) > 0]
    # if tie: return 0 (conservative)
    ones = sum(block_votes)
    zeros = len(block_votes) - ones
    return 1 if ones > zeros else 0


def parse_bits(key: str, n_cycles: int):
    """
    Qiskit bitstring ordering can vary. In your current pipeline, the previous script worked
    with the assumption: bits = d(2) + s(n_cycles).
    We'll use the same assumption to stay consistent.

    Returns (d_bits, s_bits)
      d_bits: length 2 string
      s_bits: length n_cycles string
    """
    bits = key.replace(" ", "")
    d_bits = bits[:2]
    s_bits = bits[2:2+n_cycles]
    return d_bits, s_bits


def apply_correction_to_data(d_bits: str, corr: int, flip_target: int = 0):
    """
    Minimal correction model: if corr==1, flip chosen data bit.
    We keep flip_target=0 for consistency with earlier experiments.
    """
    d1, d0 = d_bits[0], d_bits[1]
    if corr == 1:
        if flip_target == 0:
            d0 = "1" if d0 == "0" else "0"
        else:
            d1 = "1" if d1 == "0" else "0"
    return d1 + d0  # keep same ordering used in previous postprocess


def score_counts(counts, n_cycles: int, decoder_fn, ideal_data: str = "00", flip_target: int = 0) -> float:
    success = 0
    total = 0
    for key, c in counts.items():
        d_bits, s_bits = parse_bits(key, n_cycles)
        corr = decoder_fn(s_bits)
        decoded = apply_correction_to_data(d_bits, corr, flip_target=flip_target)
        if decoded == ideal_data:
            success += c
        total += c
    return success / total if total else 0.0


def main():
    os.makedirs("results", exist_ok=True)

    shots = 8000
    seed = 11

    core_p = CoreParams(theta=0.20, n_cycles=20, idle_ticks_data=1, idle_ticks_anc=1)

    base_noise = NoiseParams(
        p1_data=0.001, p1_anc=0.001,
        p2_data_data=0.002, p2_data_anc=0.005,
        pid_data=0.001, pid_anc=0.001,
        ro_anc=0.01
    )

    # Use the same circuit as S2(A): log syndrome, NO in-circuit feedback
    # Keep uncompute=False because it empirically gave the best logging/retention behavior.
    hook_log = make_topo_hook(TopoMinimalParams(check_reps=1, uncompute=False, feedback=False, correct_target=0))

    link_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    Ws = [1, 2, 4, 5, 10]  # block sizes to test (W=1 = per-cycle vote; W=20 ~ flat-ish)

    rows = []

    for m in link_mults:
        noise_m = NoiseParams(
            p1_data=base_noise.p1_data, p1_anc=base_noise.p1_anc,
            p2_data_data=base_noise.p2_data_data,
            p2_data_anc=min(0.45, base_noise.p2_data_anc * m),
            pid_data=base_noise.pid_data, pid_anc=base_noise.pid_anc,
            ro_anc=base_noise.ro_anc
        )

        nm = build_noise_model(noise_m)
        sim = AerSimulator(method="automatic", noise_model=nm, seed_simulator=seed)

        qc = build_core_circuit_with_syndrome(core_p, hook=hook_log, n_cycles=core_p.n_cycles)
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts(qc)

        # baseline: flat majority (S2(A))
        s2_flat = score_counts(
            counts, core_p.n_cycles,
            decoder_fn=lambda s: decode_flat_majority(s),
            ideal_data="00",
            flip_target=0
        )

        # S3: block consensus for various W
        s3_scores = {}
        for W in Ws:
            s3_scores[W] = score_counts(
                counts, core_p.n_cycles,
                decoder_fn=lambda s, W=W: decode_block_consensus(s, W),
                ideal_data="00",
                flip_target=0
            )

        rows.append((m, s2_flat, *[s3_scores[W] for W in Ws]))
        print(f"link_mult={m:>5} | S2_flat={s2_flat:.6f} | " +
              " ".join([f"S3_W{W}={s3_scores[W]:.6f}" for W in Ws]))

    # save csv
    out_csv = os.path.join("results", "s3_consensus.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult", "S2_flat_majority"] + [f"S3_block_consensus_W{W}" for W in Ws])
        w.writerows(rows)

    # plot
    xs = [r[0] for r in rows]
    plt.figure()
    plt.plot(xs, [r[1] for r in rows], marker="o", label="S2(A): flat majority")
    for i, W in enumerate(Ws):
        plt.plot(xs, [r[2+i] for r in rows], marker="o", label=f"S3: block-consensus (W={W})")
    plt.xscale("log")
    plt.xlabel("data-ancilla link noise multiplier (log scale)")
    plt.ylabel("decoded success probability")
    plt.title("Scenario 3 (MVP): blockchain-like block consensus decoder")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join("results", "s3_consensus.png")
    plt.savefig(out_png, dpi=200)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
