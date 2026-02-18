from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List

COMMON_RESULT_COLUMNS: List[str] = [
    "experiment",
    "mode",
    "seed",
    "shots",
    "max_rounds",
    "gamma_x",
    "chi",
    "sigma_z",
    "zeta",
    "eta_t",
    "t_m",
    "t_r",
    "tau_int",
    "p_m",
    "p_r",
    "alpha_pr_over_pm",
    "k",
    "k_star",
    "k_star_naive",
    "lifetime_time",
    "lifetime_time_median",
    "lifetime_time_ci_low",
    "lifetime_time_ci_high",
    "lifetime_rounds",
    "logical_failure_prob",
    "survival_prob",
    "avg_measurements_to_fail",
    "measurement_efficiency",
    "gain_vs_full",
    "measurement_gain_vs_full",
    "gamma_lphi",
    "gamma_lphi_ci_low",
    "gamma_lphi_ci_high",
    "gamma_lphi_ratio_vs_full",
    "safe_zone",
    "coherence_time",
    "coherence_value",
    "notes",
]


def make_result_row(**kwargs) -> Dict[str, object]:
    row: Dict[str, object] = {col: "" for col in COMMON_RESULT_COLUMNS}
    for key, value in kwargs.items():
        if key not in row:
            raise KeyError(f"Unknown result column: {key}")
        row[key] = value
    return row


def write_result_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMMON_RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COMMON_RESULT_COLUMNS})


def logspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [float(start)]
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("logspace requires start>0 and stop>0.")
    import math

    a = math.log10(float(start))
    b = math.log10(float(stop))
    step = (b - a) / float(num - 1)
    return [10.0 ** (a + i * step) for i in range(num)]


def _fmt_num(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.4g}".replace(".", "p")
    return str(x)


def build_result_path(experiment: str, *, ext: str = "csv", **params: object) -> str:
    parts = [experiment]
    for k in sorted(params.keys()):
        v = params[k]
        if v is None or v == "":
            continue
        parts.append(f"{k}-{_fmt_num(v)}")
    name = "_".join(parts) + f".{ext}"
    return os.path.join("results", name)
