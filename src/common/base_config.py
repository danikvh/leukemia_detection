"""
Base Configuration Module

Contains common configuration that's shared across all components.
"""

import os
import re
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseConfig(ABC):
    """Base configuration class with common functionality."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize with base configuration."""
        self._config = self._get_default_config()
        
        if config_file:
            self.load_from_file(config_file)
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration. Must be implemented by subclasses."""
        pass
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from file."""
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            self._update_config(self._config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise
    
    def _update_config(self, base_config: Dict[str, Any], file_config: Dict[str, Any]) -> None:
        """Recursively update configuration dictionary."""
        for key, value in file_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to file."""
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration file: {e}")
            raise
    
    def __getattr__(self, name):
        """Get configuration value with dot notation support."""
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"'BaseConfig' object has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def __setattr__(self, name, value):
        # Redirect all attribute sets to self._config if attribute exists in config keys
        # Allow normal setting for _config and other private attributes
        if name == '_config' or name.startswith('_'):
            super().__setattr__(name, value)
        elif '_config' in self.__dict__ and name in self._config:
            self._config[name] = value
        else:
            super().__setattr__(name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self._config, indent=2)})"

    def __str__(self):
        """Pretty YAML-like output for readability"""
        return yaml.dump(self._config, default_flow_style=False)