from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | os.PathLike[str], obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data


def dump_yaml(path: str | os.PathLike[str], data: Dict[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


@dataclass
class RunPaths:
    run_dir: str
    config_resolved_path: str
    metrics_path: str


def create_run_dir(
    runs_root: str | os.PathLike[str],
    exp_name: str,
    *,
    config_path: Optional[str] = None,
) -> RunPaths:
    runs_root = str(runs_root)
    ensure_dir(runs_root)
    run_dir = os.path.join(runs_root, f"{now_compact()}__{exp_name}")
    ensure_dir(run_dir)

    resolved = os.path.join(run_dir, "config_resolved.yaml")
    metrics = os.path.join(run_dir, "metrics.json")

    # Optional: record the hash of the input config for traceability.
    if config_path and os.path.exists(config_path):
        save_json(
            os.path.join(run_dir, "meta.json"),
            {"config_path": config_path, "config_sha256": sha256_file(config_path)},
        )

    return RunPaths(run_dir=run_dir, config_resolved_path=resolved, metrics_path=metrics)


def as_dict(obj: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported config object type: {type(obj)}")

