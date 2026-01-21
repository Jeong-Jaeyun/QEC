from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
from ruamel.yaml import YAML

# YAML loader: prefer PyYAML, fallback to ruamel.yaml
def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml  # PyYAML
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping/dict: {path}")
        return data
    except ImportError:
        # fallback: ruamel.yaml
        y = YAML(typ="safe")
        with open(path, "r", encoding="utf-8") as f:
            data = y.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping/dict: {path}")
        return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base (base not mutated).
    - dict vs dict: recursive
    - otherwise: override wins
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_path: Optional[str], override_path: Optional[str]) -> Dict[str, Any]:
    """
    Load base yaml then overlay scenario yaml.
    If base_path is None, try config/base.yaml if exists.
    """
    cfg: Dict[str, Any] = {}

    if base_path is None:
        default_base = os.path.join("config", "base.yaml")
        if os.path.exists(default_base):
            base_path = default_base

    if base_path:
        cfg = deep_merge(cfg, _load_yaml(base_path))

    if override_path:
        cfg = deep_merge(cfg, _load_yaml(override_path))

    return cfg


def cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    path: "core.theta" 같은 dot path
    """
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def apply_cfg_to_args(args, cfg: Dict[str, Any]) -> None:
    """
    argparse args 객체에 cfg 값을 '존재할 때만' 주입한다.
    args에 속성이 없으면 무시한다.
    """
    mapping = {
        # run
        "seed": ("run.seed", None),
        "shots": ("run.shots", None),
        "mode": ("run.mode", None),

        # core
        "theta": ("core.theta", None),
        "n_cycles": ("core.n_cycles", None),
        "idle_data": ("core.idle_ticks_data", None),
        "idle_anc": ("core.idle_ticks_anc", None),

        # noise
        "p1_data": ("noise.p1_data", None),
        "p1_anc": ("noise.p1_anc", None),
        "p2_data_data": ("noise.p2_data_data", None),
        "p2_data_anc": ("noise.p2_data_anc", None),
        "pid_data": ("noise.pid_data", None),
        "pid_anc": ("noise.pid_anc", None),
        "ro_anc": ("noise.ro_anc", None),

        # decoded
        "ideal": ("decode.ideal_data", None),
        "flip_target": ("decode.flip_target", None),
        "block_W": ("decode.block_W", None),

        # sweep
        "link_mults": ("sweep.link_mults", None),
    }

    for attr, (cfg_path, _) in mapping.items():
        if not hasattr(args, attr):
            continue
        v = cfg_get(cfg, cfg_path, default=None)
        if v is not None:
            setattr(args, attr, v)
