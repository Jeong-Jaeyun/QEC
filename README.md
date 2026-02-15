# QEC (3-Qubit Repetition / Bit-Flip Code)

This repository implements the **3-qubit repetition code** to protect a single logical qubit from **bit-flip (X) errors**:

$$|\psi\\rangle_L = \\alpha|000\\rangle + \\beta|111\\rangle.$$

## What It Does

- **Encoding:** CNOT-based encoding of `q0` into data qubits `(q0,q1,q2)`.
- **Syndrome extraction (repeated):** measures the two Z-stabilizers using a single reusable ancilla `q3`:
  - $Z_0 Z_1$
  - $Z_1 Z_2$
- **Recovery (demo):** post-processing decoder that uses the repeated syndrome bits to infer a single-qubit X correction, then decodes the logical Z-basis bit by majority vote.

## Quickstart

Noiseless demo (works without `qiskit-aer`):

```powershell
python main.py --logical_state 0 --n_cycles 3 --shots 2000
```

Deterministic single-error injection (shows syndrome -> correction):

```powershell
python main.py --logical_state 0 --n_cycles 1 --inject_x --inject_target 0 --inject_cycle 0
```

Noise sweeps and plots (requires `qiskit-aer` for noise models):

```powershell
python experiments/compare.py --mode decoded --noise
```

## Dependencies

- Required: `qiskit`
- Also required (usually pulled in automatically): `typing_extensions>=4.0`  
  If you see `ImportError: cannot import name 'Self' from 'typing_extensions'`, upgrade it with:
  `python -m pip install -U typing_extensions`
- Optional (recommended): `qiskit-aer` (for noise models and density-matrix simulations)
- Optional: `matplotlib` (for `experiments/compare.py` plots)
