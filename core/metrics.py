from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from core.circuit import encode_repetition_3, prepare_logical_state
from utils.logging import parse_syndromes_from_counts


def initial_encoded_density_matrix(logical_state: str = "+") -> DensityMatrix:
    """
    Ideal encoded 3-qubit repetition-code state on 3 data qubits (no ancilla).
    """
    qc = QuantumCircuit(3)
    prepare_logical_state(qc, logical_state)
    encode_repetition_3(qc)
    sv = Statevector.from_instruction(qc)
    return DensityMatrix(sv)


def data_fidelity_from_density_matrix(
    rho_final: DensityMatrix,
    *,
    data_qubits: tuple[int, ...] = (0, 1, 2),
    logical_state: str = "+",
) -> float:
    """
    Compute fidelity between a final multi-qubit state and the ideal encoded data state.
    - If rho_final includes extra ancilla qubits, we trace them out.
    """
    n = rho_final.num_qubits
    keep = set(data_qubits)
    traced_out = [q for q in range(n) if q not in keep]
    rho_data_final = partial_trace(rho_final, traced_out) if traced_out else rho_final
    rho_data_init = initial_encoded_density_matrix(logical_state=logical_state)
    return float(state_fidelity(DensityMatrix(rho_data_final), rho_data_init))


def syndrome_stats(counts: dict, n_cycles: int) -> dict:
    """
    Simple observability stats from get_counts() output.
    - detection_rate: P(any syndrome bit == 1) over the whole logged sequence
    - false_negative: P(all syndrome bits == 0)
    """
    syndromes = parse_syndromes_from_counts(counts)
    total = len(syndromes)
    if total == 0:
        return {"detection_rate": 0.0, "false_negative": 0.0}

    zeros = "0" * (2 * int(n_cycles))
    detected = sum(1 for s in syndromes if ("1" in s))
    fn = sum(1 for s in syndromes if s == zeros)

    return {
        "detection_rate": detected / total,
        "false_negative": fn / total,
    }

