import os
from typing import Any, Dict

import yaml


def _project_root() -> str:
    """
    Return the absolute path to the project root (parent of src/).
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_experiment_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML experiment configuration relative to the project root.
    """
    if os.path.isabs(path):
        full_path = path
    else:
        full_path = os.path.join(_project_root(), path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


