from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit import QuantumCircuit
from .circuit import prepare_initial_state

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
