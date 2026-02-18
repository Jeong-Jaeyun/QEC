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

## Effective-Model Experiments (QEC_develop.md Spec)

The repository also includes an effective Lindbladian research track aligned with `QEC_develop.md`:

- Baselines fixed to 3 modes:
  - `full_syndrome` (`k=1`)
  - `naive_sparse` (`k>1`, no autonomous correction)
  - `autonomous_sparse` (`k>1`, with autonomous correction)
- Dimensionless controls:
  - `chi = Gamma_eff * tau_int`
  - `zeta = sigma_z * tau_int`
  - `eta_t = (t_m + t_r) / tau_int`
- Core metrics:
  - logical lifetime per time
  - measurement-normalized gain
  - logical dephasing rate `Gamma_{L,phi}` from `C_L(t)=<X1X2X3>`

Run the four experiment modules:

```powershell
python experiments/phase_diagram_kstar.py
python experiments/phase_diagram_gain.py
python experiments/phase_diagram_pmpr.py
python experiments/safe_zone.py
python experiments/coherence_experiment.py
```

Checklist alias entry-points are also available:
- `python experiments/phase_kstar.py`
- `python experiments/phase_gain.py`
- `python experiments/phase_pmpr.py`

All experiment scripts above write CSV files using one shared schema via `experiments/_result_schema.py`.
Schema reference: `results/schema.md`.

Common CLI aliases are now unified across the effective-model experiments:
- `--shots` / `--trajectories`
- `--max_rounds` / `--n_rounds_max`
- `--t_m` / `--tm`, `--t_r` / `--tr`
- `--k_sparse` / `--k` (where sparse mode applies)

Optional AER quasi-static drift validation track (`QEC_develop.md` Option A):

```powershell
python experiments/aer_quasistatic_batch.py --noise --logical_state 0 --n_rounds 8 --k 4 --shots_total 8000 --batch_M 32 --sigma_z 0.03 --tau_int 1.0 --p_m 0.01 --p_r 0.01
```

(`--qsd_sigma` is supported as an alias of `--sigma_z`.)

This writes:
- `<out_prefix>_batches.csv` (per-batch `delta_i`, `theta_i`, `shots_i`, `P_i`)
- `<out_prefix>_summary.csv` (weighted ensemble average and sensitivity summary)

Measurement convention is aligned with effective-model:
- first sparse syndrome measurement occurs after `k` interactions.
- for 0-based round index `r`, measurement condition is `(r+1) % k == 0`.

One-command cross-track run (AER + effective model with matched sparse schedule and shared parameter set):

```powershell
python experiments/compare_tracks.py --logical_state 0 --n_rounds 8 --k 4 --sigma_z 0.03 --tau_int 1.0 --eta_t 1.0 --p_m 0.01 --p_r 0.01 --mode naive_sparse
```

This writes:
- `results/..._aer_summary.csv`
- `results/..._compare.csv`
- `results/..._effective_stats.csv`

By default wrapper uses `--aer_noise` (requires `qiskit-aer`). Use `--no_aer_noise` for a BasicSimulator smoke run.

Implementation map for the AQEC research track:
- Effective autonomous correction + sparse monitoring + Bayes/HMM-style filtering:
  - `core/effective_model.py`
- Quasistatic Z-drift and coherence extraction (`Gamma_{L,phi}`):
  - `core/effective_model.py`, `experiments/coherence_experiment.py`
- Safe-zone criterion (`lifetime gain` + `dephasing ratio`):
  - `experiments/safe_zone.py`

## Dependencies

- Effective-model only: install `requirements-effective.txt`
- Circuit/AER track: install `requirements-aer.txt`
- Also required (usually pulled in automatically): `typing_extensions>=4.0`
  If you see `ImportError: cannot import name 'Self' from 'typing_extensions'`, upgrade it with:
  `python -m pip install -U typing_extensions`

Test behavior without `qiskit`:
- `tests/test_nonlocal_buffer.py` is auto-skipped.
- `tests/test_repetition_code.py` keeps pure-python decoder tests active and skips only circuit-shape checks.
