# src/core/config.py
"""Configuration loader."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and merges configuration from YAML files."""

    @staticmethod
    def load(config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to config file. If None, loads default.yaml

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Try to find default config
            possible_paths = [
                Path("config/default.yaml"),
                Path(__file__).parent.parent.parent / "config" / "default.yaml",
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

            if config_path is None:
                raise FileNotFoundError("Could not find default configuration file")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def save(config: Dict[str, Any], config_path: Path):
        """Save configuration to file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
