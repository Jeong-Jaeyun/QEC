# Result Schema

Shared CSV schema used by:
- `experiments/phase_diagram_kstar.py`
- `experiments/phase_diagram_gain.py`
- `experiments/phase_diagram_pmpr.py`
- `experiments/safe_zone.py`
- `experiments/coherence_experiment.py`

Cross-track wrapper:
- `experiments/compare_tracks.py` writes dedicated comparison CSV files:
  - `*_compare.csv` (AER vs effective side-by-side metrics)
  - `*_effective_stats.csv` (serialized `LifetimeStats`)
  - `*_aer_summary.csv` (from `aer_quasistatic_batch.py`)

Core fields:
- `experiment`, `mode`, `seed`, `shots`, `max_rounds`
- `gamma_x`, `chi`, `sigma_z`, `zeta`
- `tau_int`, `eta_t`, `t_m`, `t_r`
- `p_m`, `p_r`, `alpha_pr_over_pm`
- `k`, `k_star`, `k_star_naive`

Lifetime / gain:
- `lifetime_time`
- `lifetime_time_median`
- `lifetime_time_ci_low`, `lifetime_time_ci_high`
- `lifetime_rounds`
- `logical_failure_prob`, `survival_prob`
- `avg_measurements_to_fail`, `measurement_efficiency`
- `gain_vs_full`, `measurement_gain_vs_full`

Coherence / safe zone:
- `gamma_lphi`
- `gamma_lphi_ci_low`, `gamma_lphi_ci_high`
- `gamma_lphi_ratio_vs_full`
- `safe_zone`
- `coherence_time`, `coherence_value`

Other:
- `notes`
