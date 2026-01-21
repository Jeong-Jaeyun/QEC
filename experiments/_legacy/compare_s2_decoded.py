import matplotlib
matplotlib.use("Agg")
import os, csv
from collections import Counter
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from core.circuit import CoreParams
from core.noise import NoiseParams, build_noise_model
from core.circuit import build_core_circuit_with_syndrome
from scenarios.topo_minimal import TopoMinimalParams, make_topo_hook
from utils.logging import score_decoded_success, decode_flat_majority


def decode_majority_syndrome(bitstring_s: str) -> int:
    """
    Simple decoder: majority vote over syndrome bits.
    Return 1 if more 1s than 0s, else 0.
    """
    ones = bitstring_s.count("1")
    zeros = len(bitstring_s) - ones
    return 1 if ones > zeros else 0

'''
def postprocess_and_score(counts, n_cycles: int, ideal_data: str = "00") -> float:
    success = 0
    total = 0

    for key, c in counts.items():
        # Qiskit bitstring uses little-endian per register, and registers reversed in concatenation.
        # Typically it is: d (2) then s (n_cycles) in the string, with spaces removed.
        bits = key.replace(" ", "")

        # assume layout: d[1]d[0] + s[n-1]...s[0]
        d_bits = bits[:2]
        s_bits = bits[2:2+n_cycles]

        # decoder suggests whether to flip one data bit (minimal correction model)
        corr = decode_majority_syndrome(s_bits)

        # apply correction: flip d0 if corr==1 (consistent with previous correct_target=0 choice)
        d1, d0 = d_bits[0], d_bits[1]
        if corr == 1:
            d0 = "1" if d0 == "0" else "0"

        decoded = d0 + d1  # choose consistent ordering for comparison
        # ideal_data must match this ordering (set to "00" for now)
        if decoded == ideal_data:
            success += c
        total += c

    return success / total if total > 0 else 0.0
'''

def main():
    os.makedirs("results", exist_ok=True)

    # Use qasm shots for syndrome-based evaluation
    shots = 8000
    seed = 11

    core_p = CoreParams(theta=0.20, n_cycles=20, idle_ticks_data=1, idle_ticks_anc=1)

    base_noise = NoiseParams(
        p1_data=0.001, p1_anc=0.001,
        p2_data_data=0.002, p2_data_anc=0.005,
        pid_data=0.001, pid_anc=0.001,
        ro_anc=0.01
    )

    # Optimal S2b structure we found (uncompute=False)
    hook_s2 = make_topo_hook(TopoMinimalParams(check_reps=1, uncompute=False, feedback=False, correct_target=0))
    # feedback=False 이유: 이제는 "측정→후처리 디코딩"이 피드백 역할을 함

    link_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
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

        qc = build_core_circuit_with_syndrome(core_p, hook=hook_s2, n_cycles=core_p.n_cycles)
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts(qc)

        score = score_decoded_success(
            counts,
            n_cycles=core_p.n_cycles,
            decoder=decode_flat_majority,
            ideal_data="00",
            flip_target=0,
        )
        rows.append((m, score))
        print(f"link_mult={m:>5} | decoded_success={score:.6f}")

    # save
    out_csv = os.path.join("results", "s2_decoded.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult", "decoded_success_prob"])
        w.writerows(rows)

    # plot
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    plt.figure()
    plt.plot(xs, ys, marker="o", label="S2 decoded (syndrome majority)")
    plt.xscale("log")
    plt.xlabel("data-ancilla link noise multiplier (log scale)")
    plt.ylabel("decoded success probability")
    plt.title("Scenario 2 (A): syndrome logging + classical post-decoder")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join("results", "s2_decoded.png")
    plt.savefig(out_png, dpi=200)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
