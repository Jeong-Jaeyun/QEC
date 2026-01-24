from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity, partial_trace
from qiskit import QuantumCircuit
from .circuit import prepare_initial_state
from utils.logging import parse_syndromes_from_counts


def initial_data_density_matrix() -> DensityMatrix:
    qc = QuantumCircuit(3)
    prepare_initial_state(qc)
    sv = Statevector.from_instruction(qc)
    rho = DensityMatrix(sv)
    rho_data = partial_trace(rho, [2])  # trace out ancilla
    return DensityMatrix(rho_data)

def data_fidelity_from_density_matrix(rho_final: DensityMatrix) -> float:
    rho_data_final = partial_trace(rho_final, [2])
    rho_data_init = initial_data_density_matrix()
    fid = state_fidelity(DensityMatrix(rho_data_final), rho_data_init)
    return float(fid)

def fidelity_data_2q(result, circuit, data_qubits=(0, 1), ideal="00"):
    """
    AerSimulator(method='density_matrix') 결과에서,
    data_qubits에 대한 reduced density matrix를 뽑아 |ideal><ideal| 과 fidelity 계산.
    """
    rho_full = result.data(0)["density_matrix"]
    rho_full = DensityMatrix(rho_full)

    # 부분계 추출
    traced_out = [q for q in range(rho_full.num_qubits) if q not in data_qubits]
    rho_red = partial_trace(rho_full, traced_out)
    
    # ideal 상태
    if ideal == "00":
        psi = Statevector.from_label("00")
    else:
        psi = Statevector.from_label(ideal)

    return float(state_fidelity(rho_red, psi))

def syndrome_stats(counts: dict, n_cycles: int) -> dict:
    syndromes = parse_syndromes_from_counts(counts)
    total = len(syndromes)
    if total == 0:
        return {"detection_rate": 0.0, "false_negative": 0.0}

    zeros = "0" * n_cycles
    detected = sum(1 for s in syndromes if ("1" in s))
    fn = sum(1 for s in syndromes if s == zeros)

    return {
        "detection_rate": detected / total,   # P(syndrome != 0)
        "false_negative": fn / total,         # P(syndrome == 0)
    }
