import matplotlib
matplotlib.use("Agg")

import os
import csv
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix

from core.circuit import CoreParams, build_core_circuit
from core.noise import NoiseParams, build_noise_model
from core.metrics import data_fidelity_from_density_matrix

from scenarios.topo_minimal import TopoMinimalParams, make_topo_hook


def run(core_p: CoreParams, noise_p: NoiseParams, hook=None, shots=4000, seed=11):
    nm = build_noise_model(noise_p)
    sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed)

    qc = build_core_circuit(core_p, hook=hook, measure_ancilla=True, reset_ancilla=True)
    qc.save_density_matrix()

    res = sim.run(qc, shots=shots).result()
    rho = DensityMatrix(res.data(qc)["density_matrix"])
    return float(data_fidelity_from_density_matrix(rho))


def main():
    os.makedirs("results", exist_ok=True)

    core_p = CoreParams(theta=0.20, n_cycles=20, idle_ticks_data=1, idle_ticks_anc=1)

    # Use the same link-noise-aware model (2Q included)
    base_noise = NoiseParams(
        p1_data=0.001, p1_anc=0.001,
        p2_data_data=0.002,
        p2_data_anc=0.005,     # link noise baseline
        pid_data=0.001, pid_anc=0.001,
        ro_anc=0.01
    )

    # Two variants:
    # S2a: constraint-only (no feedback)
    hook_s2a = make_topo_hook(TopoMinimalParams(check_reps=1, uncompute=True, feedback=False, correct_target=0))

    # S2b: constraint + minimal feedback
    hook_s2b = make_topo_hook(TopoMinimalParams(check_reps=1, uncompute=True, feedback=True, correct_target=0))

    # Sweep link noise multiplier (same idea as S1) — topo-like schemes should show different sensitivity
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

        fid_base = run(core_p, noise_m, hook=None)
        fid_s2a  = run(core_p, noise_m, hook=hook_s2a)
        fid_s2b  = run(core_p, noise_m, hook=hook_s2b)

        rows.append((m, fid_base, fid_s2a, fid_s2b))
        print(f"link_mult={m:>5} | base={fid_base:.6f} | S2a(no fb)={fid_s2a:.6f} | S2b(fb)={fid_s2b:.6f}")

    # Save CSV
    csv_path = os.path.join("results", "s2_compare.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link_mult", "fidelity_base", "fidelity_s2a_constraint_only", "fidelity_s2b_with_feedback"])
        w.writerows(rows)

    # Plot
    xs = [r[0] for r in rows]
    y0 = [r[1] for r in rows]
    y1 = [r[2] for r in rows]
    y2 = [r[3] for r in rows]

    plt.figure()
    plt.plot(xs, y0, marker="o", label="S0_BASE (no hook)")
    plt.plot(xs, y1, marker="o", label="S2a_TOPO_MIN (constraint only)")
    plt.plot(xs, y2, marker="o", label="S2b_TOPO_MIN (constraint + feedback)")
    plt.xscale("log")
    plt.xlabel("data-ancilla link noise multiplier (log scale)")
    plt.ylabel("data fidelity (q0,q1)")
    plt.title("Scenario 2 MVP: constraint extraction vs minimal feedback")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join("results", "s2_compare.png")
    plt.savefig(fig_path, dpi=200)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
