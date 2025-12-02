"""Configuration utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert string numbers to proper types (handles scientific notation)
    config = _convert_numeric_strings(config)
    
    return config


def _convert_numeric_strings(config: Any) -> Any:
    """Recursively convert numeric strings to proper types."""
    if isinstance(config, dict):
        return {k: _convert_numeric_strings(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_convert_numeric_strings(v) for v in config]
    elif isinstance(config, str):
        # Try to convert scientific notation strings (e.g., "1e-4")
        import re
        # Match scientific notation pattern: optional sign, digits, optional decimal, optional exponent
        if re.match(r'^-?\d+\.?\d*[eE][+-]?\d+$', config):
            try:
                return float(config)
            except ValueError:
                pass
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to {save_path}")


def print_config(config: Dict[str, Any], indent: int = 0):
    """Print configuration in a formatted way."""
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

