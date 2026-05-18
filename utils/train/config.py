"""
Configuration classes with dynamic attributes from YAML/JSON.
"""

from typing import Dict, Any
import yaml
import json


_MODEL_NESTED = ('ccc', 'feat', 'graph', 'tree')
_TRAING_NESTED = ('ccc', 'feat')


def _apply_kwargs(inst, nested_keys: tuple, nested_cls: type, kwargs: Dict) -> None:
    """Set attributes from kwargs; dict values for nested_keys become nested_cls instances."""
    for k in nested_keys:
        setattr(inst, k, None)
    for key, value in kwargs.items():
        if key in nested_keys and isinstance(value, dict):
            setattr(inst, key, nested_cls(**value))
        else:
            setattr(inst, key, value)


class ModelConfig:
    """Model architecture config; nested keys: ccc, feat, graph, tree."""
    def __init__(self, **kwargs):
        _apply_kwargs(self, _MODEL_NESTED, ModelConfig, kwargs)

    def __getattr__(self, name):
        return None


class TraingConfig:
    """Training config; nested keys: ccc, feat."""
    def __init__(self, **kwargs):
        _apply_kwargs(self, _TRAING_NESTED, TraingConfig, kwargs)

    def __getattr__(self, name):
        return None


class ExperimentConfig:
    """
    Main configuration class for CellSTIC.
    Supports dynamic attributes from YAML files.
    """
    def __init__(self, **kwargs):
        # Initialize sub-configurations
        self.model = ModelConfig()
        self.train = TraingConfig()
    
    def __getattr__(self, name):
        # Return None for undefined attributes
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)


def load_config(filepath: str, format: str = "yaml") -> ExperimentConfig:
    """
    Load configuration from file with full dynamic attribute support.
    
    Args:
        filepath: Path to configuration file
        format: File format ("yaml", "json")
        
    Returns:
        CellSTIC configuration object with all YAML attributes dynamically loaded
    """
    with open(filepath, 'r') as f:
        if format.lower() == "yaml":
            data = yaml.safe_load(f)
        elif format.lower() == "json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # Create config with dynamic loading
    config = ExperimentConfig()
    
    # Process each top-level key
    for key, value in data.items():
        if key in ['model', 'train']:
            # Handle nested configurations
            nested_config = getattr(config, key)
            if isinstance(value, dict):
                # Update the nested config with the dictionary values
                nested_config.__init__(**value)
        else:
            # Handle top-level attributes (including custom ones)
            setattr(config, key, value)
    
    return config
