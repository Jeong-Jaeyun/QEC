import matplotlib
matplotlib.use("Agg")

import os, csv
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix

from core.circuit import CoreParams, build_core_circuit
from core.noise import NoiseParams, build_noise_model
from core.metrics import data_fidelity_from_density_matrix

from scenarios.topo_minimal import TopoMinimalParams, make_topo_hook


def run(core_p, noise_p, hook=None, shots=4000, seed=11):
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
    base_noise = NoiseParams(
        p1_data=0.001, p1_anc=0.001,
        p2_data_data=0.002, p2_data_anc=0.005,
        pid_data=0.001, pid_anc=0.001,
        ro_anc=0.01
    )

    # 빠른 랭킹용 link_mult 샘플 (필요하면 늘려도 됨)
    link_mults = [1.0, 3.0, 8.0, 12.0]

    reps_list = [1, 2, 3]
    uncompute_list = [True, False]
    target_list = [0, 1]

    rows = []
    for reps in reps_list:
        for unc in uncompute_list:
            for tgt in target_list:
                hook = make_topo_hook(TopoMinimalParams(
                    check_reps=reps,
                    uncompute=unc,
                    feedback=True,
                    correct_target=tgt
                ))

                fids = []
                for m in link_mults:
                    noise_m = NoiseParams(
                        p1_data=base_noise.p1_data, p1_anc=base_noise.p1_anc,
                        p2_data_data=base_noise.p2_data_data,
                        p2_data_anc=min(0.45, base_noise.p2_data_anc * m),
                        pid_data=base_noise.pid_data, pid_anc=base_noise.pid_anc,
                        ro_anc=base_noise.ro_anc
                    )
                    f = run(core_p, noise_m, hook=hook)
                    fids.append(f)

                avg_fid = float(np.mean(fids))
                # 간단 점수: 평균 fidelity (원하면 가중치/면적(AUC)로 바꿀 수 있음)
                rows.append((reps, unc, tgt, avg_fid, *fids))
                print(f"reps={reps} uncompute={unc} tgt={tgt} | avg={avg_fid:.6f} | fids={['%.4f'%x for x in fids]}")

    # 평균 fidelity 기준으로 정렬
    rows.sort(key=lambda x: x[3], reverse=True)

    out_csv = os.path.join("results", "s2_sweep_rank.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check_reps", "uncompute", "correct_target", "avg_fidelity"] + [f"fid_m{m}" for m in link_mults])
        w.writerows(rows)

    print(f"\nSaved: {out_csv}")
    print("Top-5 configs:")
    for r in rows[:5]:
        print(r[:4])


if __name__ == "__main__":
    main()
