"""
RAG-Benchmark Hauptprojekt

Author: Lukas Schaumlöffel
Master Informatik (HAW Hamburg)
"""

__version__ = "1.0.0"
__author__ = "Lukas Schaumlöffel"

# Main module imports for convenience
from .data.loader import DataLoader
from .validation.config_validator import ConfigValidator, validate_and_load_config

__all__ = [
    'DataLoader',
    'ConfigValidator',
    'validate_and_load_config'
]