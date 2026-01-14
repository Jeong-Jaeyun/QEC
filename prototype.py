from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit.circuit.library import UnitaryGate
import copy
import matplotlib.pyplot as plt

# Configuration

@dataclass
class BaseParams:
    # Three-body coupling strength absorbed into theta; theta = J * t
    theta: float = 0.20

    # Cycles: (bind -> noise -> measure ancilla) repeated
    n_cycles: int = 20
    shots: int = 4000

    # Baseline noise rates (tune later)
    p1: float = 0.001   # 1Q depolarizing
    p2: float = 0.005   # 2Q depolarizing (used if you add 2Q gates)
    pm: float = 0.010   # measurement depolarizing (approx)

    # Memory / idle noise approximation per cycle via id gate depolarizing
    pid_data: float = 0.001
    pid_anc: float = 0.001

    # Attack knobs (scenario dependent)
    attack_on_anc: float = 0.0  # extra depolarizing on ancilla (buffer)
    attack_on_data: float = 0.0 # extra depolarizing on data

    # "Blockchain-like" consensus window (scenario 3)
    consensus_k: int = 5


@dataclass
class ScenarioConfig:
    name: str  # "S0_BASE", "S1_NONLOCAL_BUFFER", "S2_TOPO_MINIMAL", "S3_BLOCKCHAIN_LIKE"
    params: BaseParams



# Three-body unitary U = exp(-i theta Z⊗Z⊗Z)


def make_zzz_unitary(theta: float) -> UnitaryGate:
    phases = []
    for b in range(8):
        b0 = (b >> 0) & 1
        b1 = (b >> 1) & 1
        b2 = (b >> 2) & 1
        parity = (b0 + b1 + b2) % 2
        eigen = +1 if parity == 0 else -1
        phases.append(np.exp(-1j * theta * eigen))
    U = np.diag(phases).astype(complex)
    return UnitaryGate(U, label="U_ZZZ")



# Noise model builder


def build_noise_model(p: BaseParams) -> NoiseModel:
    nm = NoiseModel()

    # 1Q gate noise (we will use 'u' gates implicitly + explicit 'id')
    e1 = depolarizing_error(p.p1, 1)
    nm.add_all_qubit_quantum_error(e1, ["u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z", "h", "s", "sdg", "t", "tdg"])

    # idle noise via 'id' gate; we will insert id gates each cycle
    # We cannot assign different idle errors by qubit via add_all_qubit... easily without per-qubit mapping,
    # so we will handle ancilla-vs-data by inserting extra "attack" id gates on target qubits.
    eid = depolarizing_error(p.pid_data, 1)
    nm.add_all_qubit_quantum_error(eid, ["id"])

    # measurement noise approx: map to readout error is possible, but keep simple by extra depolarizing before measure
    # We'll insert a 1Q depolarizing gate-equivalent via an 'id' with higher pid if needed.

    return nm


# Circuit templates

def prepare_initial_state(circ: QuantumCircuit) -> None:
    # Example target on data: Bell-like |Φ+> = (|00> + |11>)/sqrt(2)
    circ.h(0)
    circ.cx(0, 1)
    # ancilla to |0> (default)


def one_cycle(circ: QuantumCircuit, p: BaseParams, scenario: str) -> None:
    """
    One cycle = bind(ZZZ) + idle + optional attack + (optional topological-like check) + measure ancilla
    """
    # 3-body binding
    circ.append(make_zzz_unitary(p.theta), [0, 1, 2])

    # Idle step approximation (all qubits)
    circ.id(0); circ.id(1); circ.id(2)

    # Scenario-specific: non-local buffer => extra idle/attack on ancilla (single choke point)
    if scenario == "S1_NONLOCAL_BUFFER":
        # represent remote latency / link errors as extra depolarizing on ancilla
        for _ in range(3):
            circ.id(2)

    # Optional attack knobs (generic)
    # we implement "attack" as extra id gates; noise model attaches depolarizing to id.
    # For stronger attacks, increase pid_anc/pid_data in params or add more ids here.
    if p.attack_on_anc > 0:
        # approximate by extra id "ticks"
        n = int(np.ceil(10 * p.attack_on_anc))
        for _ in range(n):
            circ.id(2)

    if p.attack_on_data > 0:
        n = int(np.ceil(10 * p.attack_on_data))
        for _ in range(n):
            circ.id(0); circ.id(1)

    # Scenario 2 (minimal topo-like): measure a parity constraint via ancilla-style interaction pattern.
    # With only 3 qubits, we can force a repeated parity-check-like structure.
    # NOTE: this is not "true topological protection"—just a minimal constraint cycle.
    if scenario == "S2_TOPO_MINIMAL":
        # Entangle ancilla with Z-parity of data, then (later) measure ancilla
        circ.cx(0, 2)
        circ.cx(1, 2)
        # uncompute is intentionally omitted to mimic continuous stabilizer extraction

    # Measure ancilla each cycle into a classical bit (append a new classical register bit on demand)
    # We will build circuit with enough classical bits beforehand, so measure into the right index elsewhere.


def build_experiment_circuit(cfg: ScenarioConfig) -> QuantumCircuit:
    p = cfg.params
    # Classical bits = n_cycles (store ancilla syndrome each cycle)
    circ = QuantumCircuit(3, p.n_cycles)

    prepare_initial_state(circ)

    for k in range(p.n_cycles):
        one_cycle(circ, p, cfg.name)
        # Optional: measurement noise approximation - add extra idle before measurement
        if p.pm > 0:
            # add one more id tick on ancilla before measurement; tune via pid_anc to reflect pm if desired
            circ.id(2)

        circ.measure(2, k)

        # Reset ancilla to reuse like a buffer (common in QEC-style cycles)
        circ.reset(2)

    return circ

# Running + Metrics


def initial_data_density_matrix() -> DensityMatrix:
    qc = QuantumCircuit(3)
    prepare_initial_state(qc)
    sv = Statevector.from_instruction(qc)
    rho = DensityMatrix(sv)
    # trace out ancilla (q2)
    rho_data = partial_trace(rho, [2])
    return DensityMatrix(rho_data)


def run_and_score(cfg: ScenarioConfig, seed: int = 7) -> Dict:
    p = cfg.params
    noise = build_noise_model(p)
    sim = AerSimulator(method="density_matrix", noise_model=noise, seed_simulator=seed)

    circ = build_experiment_circuit(cfg)

    # Save final density matrix to compute fidelity on data qubits
    circ.save_density_matrix()

    result = sim.run(circ, shots=p.shots).result()
    counts = result.get_counts(circ)

    rho_final = result.data(circ)["density_matrix"]
    rho_final = DensityMatrix(rho_final)
    rho_data_final = partial_trace(rho_final, [2])

    rho_data_init = initial_data_density_matrix()
    fid = state_fidelity(DensityMatrix(rho_data_final), rho_data_init)

    # Blockchain-like "consensus" metric (scenario 3): majority over last k syndromes
    consensus = None
    if cfg.name == "S3_BLOCKCHAIN_LIKE":
        # Build marginal distribution of last k measurement bits (classical post-processing)
        k = max(1, p.consensus_k)
        consensus = consensus_score_from_counts(counts, k)

    return {
        "scenario": cfg.name,
        "params": asdict(p),
        "fidelity_data": float(np.real(fid)),
        "counts": counts,
        "consensus_score": consensus,
    }


def consensus_score_from_counts(counts: Dict[str, int], k: int) -> float:
    """
    Treat syndrome history bits as a chain. Define 'agreement' as majority of last k bits.
    Score = P(majority==0) - P(majority==1)  (example; you can redefine later)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    score = 0.0
    for bitstring, c in counts.items():
        # Qiskit count keys are like '0101...' with leftmost = highest classical bit.
        # Our measurement order is (cycle 0 -> c0, ...), but the printed string is reversed.
        bits = bitstring[::-1]  # now index 0 = cycle 0
        tail = bits[max(0, len(bits) - k):]
        ones = sum(1 for b in tail if b == "1")
        maj = 1 if ones > (len(tail) / 2) else 0
        score += (1.0 if maj == 0 else -1.0) * (c / total)
    return score


# 6Scenario presets

def make_scenarios() -> List[ScenarioConfig]:
    base = BaseParams()

    s0 = ScenarioConfig("S0_BASE", base)

    # Scenario 1: non-local buffer -> higher ancilla vulnerability / latency ticks
    p1 = BaseParams(**asdict(base))
    p1.pid_data = 0.001
    p1.pid_anc = 0.001
    # you can also push "attack_on_anc" up to stress-test
    p1.attack_on_anc = 0.3
    s1 = ScenarioConfig("S1_NONLOCAL_BUFFER", p1)

    # Scenario 2: topo-minimal -> repeated parity-constraint extraction
    p2 = BaseParams(**asdict(base))
    p2.attack_on_anc = 0.1
    s2 = ScenarioConfig("S2_TOPO_MINIMAL", p2)

    # Scenario 3: blockchain-like -> we keep quantum part similar to base,
    # but emphasize logging + consensus metric in post-processing
    p3 = BaseParams(**asdict(base))
    p3.consensus_k = 7
    s3 = ScenarioConfig("S3_BLOCKCHAIN_LIKE", p3)

    return [s0, s1, s2, s3]


def clone_scenario(sc: ScenarioConfig) -> ScenarioConfig:
    # deep-ish copy for dataclasses
    return ScenarioConfig(name=sc.name, params=BaseParams(**asdict(sc.params)))


def set_param(sc: ScenarioConfig, key: str, value):
    setattr(sc.params, key, value)


def sweep_param(
    scenarios: List[ScenarioConfig],
    param_name: str,
    values: List[float],
    seed: int = 11
) -> List[Dict]:
    """
    Returns a list of rows: {param_name, scenario, fidelity_data, consensus_score}
    """
    rows = []
    for v in values:
        for sc in scenarios:
            sc2 = clone_scenario(sc)
            set_param(sc2, param_name, v)
            out = run_and_score(sc2, seed=seed)
            rows.append({
                param_name: v,
                "scenario": out["scenario"],
                "fidelity_data": out["fidelity_data"],
                "consensus_score": out["consensus_score"],
            })
        print(f"[sweep] {param_name}={v} done")
    return rows


def plot_sweep(rows: List[Dict], x_key: str, title: str):
    # group by scenario
    scenarios = sorted(set(r["scenario"] for r in rows))
    for sc in scenarios:
        xs = [r[x_key] for r in rows if r["scenario"] == sc]
        ys = [r["fidelity_data"] for r in rows if r["scenario"] == sc]
        plt.plot(xs, ys, marker="o", label=sc)
    plt.xlabel(x_key)
    plt.ylabel("data fidelity (q0,q1)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_table(rows: List[Dict], x_key: str):
    # simple console table
    scenarios = sorted(set(r["scenario"] for r in rows))
    header = f"{x_key:>10} | " + " | ".join([f"{sc:>18}" for sc in scenarios])
    print("\n" + header)
    print("-" * len(header))
    for v in sorted(set(r[x_key] for r in rows)):
        line = f"{v:10.4g} | "
        for sc in scenarios:
            match = [r for r in rows if r[x_key] == v and r["scenario"] == sc]
            if not match:
                line += f"{'':>18} | "
            else:
                line += f"{match[0]['fidelity_data']:.6f}".rjust(18) + " | "
        print(line)


def run_sweeps():
    base_scenarios = make_scenarios()

    # Sweep A: 1Q depolarizing p1

    p1_values = [0.0, 0.001, 0.003, 0.01, 0.03, 0.05]
    rows_a = sweep_param(base_scenarios, "p1", p1_values, seed=11)
    print_table(rows_a, "p1")
    plot_sweep(rows_a, "p1", "Sweep A: fidelity vs p1 (1Q depolarizing)")

    # Sweep B: buffer attack strength (ancilla only)
    # Focus: S0 vs S1 gap emergence

    s0 = [sc for sc in base_scenarios if sc.name == "S0_BASE"][0]
    s1 = [sc for sc in base_scenarios if sc.name == "S1_NONLOCAL_BUFFER"][0]
    pair = [s0, s1]

    attack_values = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
    rows_b = sweep_param(pair, "attack_on_anc", attack_values, seed=11)
    print_table(rows_b, "attack_on_anc")
    plot_sweep(rows_b, "attack_on_anc", "Sweep B: fidelity vs attack_on_anc (S0 vs S1)")

    # Optional: show S3 consensus behavior under higher noise (if you want)
    # Example: sweep p1 for S3 only and print consensus_score
    s3 = [sc for sc in base_scenarios if sc.name == "S3_BLOCKCHAIN_LIKE"][0]
    rows_c = sweep_param([s3], "p1", p1_values, seed=11)
    print("\n[S3 consensus_score vs p1]")
    for r in rows_c:
        print(f"p1={r['p1']:.4g} consensus_score={r['consensus_score']}")

"""
if __name__ == "__main__":
    scenarios = make_scenarios()
    for sc in scenarios:
        out = run_and_score(sc, seed=11)
        print(f"\n=== {out['scenario']} ===")
        print("fidelity_data:", out["fidelity_data"])
        if out["consensus_score"] is not None:
            print("consensus_score:", out["consensus_score"])
        # Show a small slice of counts
        top = sorted(out["counts"].items(), key=lambda x: -x[1])[:5]
        print("top_counts:", top)
"""
if __name__ == "__main__":
    run_sweeps()