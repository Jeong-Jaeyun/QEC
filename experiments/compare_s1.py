import matplotlib
matplotlib.use("Agg")  # avoid tkinter main-loop issues

import os
import csv
from dataclasses import asdict
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix

from core.circuit import CoreParams, build_core_circuit
from core.noise import NoiseParams, build_noise_model
from core.metrics import data_fidelity_from_density_matrix

from scenarios.nonlocal_buffer import NonLocalBufferParams, make_nonlocal_buffer_hook


def run(core_p: CoreParams, noise_p: NoiseParams, hook=None, shots=4000, seed=11):
    nm = build_noise_model(noise_p)
    sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed)

    qc = build_core_circuit(core_p, hook=hook, measure_ancilla=True, reset_ancilla=True)
    qc.save_density_matrix()

    res = sim.run(qc, shots=shots).result()
    rho = DensityMatrix(res.data(qc)["density_matrix"])
    fid = data_fidelity_from_density_matrix(rho)
    return float(fid)


def main():
    os.makedirs("results", exist_ok=True)

    core_p = CoreParams(theta=0.20, n_cycles=20, idle_ticks_data=1, idle_ticks_anc=1)

    # Baseline noise: include NEW p2 terms
    base_noise = NoiseParams(
        p1_data=0.001, p1_anc=0.001,
        p2_data_data=0.002,
        p2_data_anc=0.005,   # baseline link error
        pid_data=0.001, pid_anc=0.001,
        ro_anc=0.01
    )

    s1p = NonLocalBufferParams(extra_anc_ticks=3, bridge_strength=1, bridge_mode="zx")
    hook = make_nonlocal_buffer_hook(s1p)

    # Sweep link fragility multiplier (apply to p2_data_anc)
    link_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

    rows = []
    for m in link_mults:
        # Baseline: no hook, baseline link noise
        fid_base = run(core_p, base_noise, hook=None)

        # S1: hook + worse data-ancilla link
        s1_noise = NoiseParams(
            p1_data=base_noise.p1_data, p1_anc=base_noise.p1_anc,
            p2_data_data=base_noise.p2_data_data,
            p2_data_anc=min(0.45, base_noise.p2_data_anc * m),  # cap for stability
            pid_data=base_noise.pid_data, pid_anc=base_noise.pid_anc,
            ro_anc=base_noise.ro_anc
        )
        fid_s1 = run(core_p, s1_noise, hook=hook)

        rows.append((m, fid_base, fid_s1))
        print(f"link_mult={m:>5} | base={fid_base:.6f} | S1={fid_s1:.6f}")
    
    # Save CSV
    csv_path = os.path.join("results", "s1_compare.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anc_strength", "fidelity_base", "fidelity_s1_nonlocal"])
        w.writerows(rows)

    # Plot
    xs = [r[0] for r in rows]
    y0 = [r[1] for r in rows]
    y1 = [r[2] for r in rows]

    plt.figure()
    plt.plot(xs, y0, marker="o", label="S0_BASE (no hook)")
    plt.plot(xs, y1, marker="o", label="S1_NONLOCAL_BUFFER (hook + ancilla fragility)")
    plt.xscale("log")
    plt.xlabel("ancilla noise multiplier (log scale)")
    plt.ylabel("data fidelity (q0,q1)")
    plt.title("Scenario 1 MVP: ancilla fragility impacts data via buffer coupling")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join("results", "s1_compare.png")
    plt.savefig(fig_path, dpi=200)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
