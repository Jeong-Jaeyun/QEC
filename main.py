import numpy as np
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from core.circuit import CoreParams, build_core_circuit
from core.noise import NoiseParams, build_noise_model
from core.metrics import data_fidelity_from_density_matrix

def run_once(core_p: CoreParams, noise_p: NoiseParams, shots: int = 2000, seed: int = 11):
    nm = build_noise_model(noise_p)
    sim = AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed)

    qc = build_core_circuit(core_p, hook=None, measure_ancilla=True, reset_ancilla=True)
    qc.save_density_matrix()

    res = sim.run(qc, shots=shots).result()
    rho = DensityMatrix(res.data(qc)["density_matrix"])
    fid = data_fidelity_from_density_matrix(rho)
    return fid

if __name__ == "__main__":
    core_p = CoreParams(theta=0.20, n_cycles=20)

    # baseline
    base = NoiseParams(p1_data=0.001, p1_anc=0.001, pid_data=0.001, pid_anc=0.001, ro_anc=0.01)
    fid0 = run_once(core_p, base)
    print("baseline fidelity:", fid0)

    # ancilla-only harsher noise (this MUST change fidelity if coupling exists)
    anc_bad = NoiseParams(p1_data=0.001, p1_anc=0.03, pid_data=0.001, pid_anc=0.03, ro_anc=0.10)
    fid1 = run_once(core_p, anc_bad)
    print("ancilla harsher fidelity:", fid1)

    # data-only harsher noise
    data_bad = NoiseParams(p1_data=0.03, p1_anc=0.001, pid_data=0.03, pid_anc=0.001, ro_anc=0.01)
    fid2 = run_once(core_p, data_bad)
    print("data harsher fidelity:", fid2)
