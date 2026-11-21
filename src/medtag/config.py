"""Config loading."""

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
