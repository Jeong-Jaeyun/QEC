# QEC
This repository implements a 3-qubit repetition code to protect quantum data from bit-flip (X) errors. It encodes one logical qubit into a three-qubit entangled state: $|\psi\rangle_L = \alpha|000\rangle + \beta|111\rangle$.
The algorithm features:Encoding: CNOT-based entanglement for redundancy.Syndrome Measurement: Uses ancilla qubits and parity checks ($Z_1Z_2, Z_2Z_3$) to detect errors without collapsing the quantum state.Recovery: Applies conditional $X$ gates based on majority voting to restore state fidelity.Ideal for demonstrating the fundamentals of fault-tolerant quantum computing.

-Own UCS LAB-
