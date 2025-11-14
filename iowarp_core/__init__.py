"""
IOWarp Core - A Comprehensive Platform for Context Management in Scientific Computing

This package provides installation and configuration utilities for IOWarp Core,
a unified framework that integrates multiple high-performance components for
context management, data transfer, and scientific computing.
"""

__version__ = "1.0.0"
__author__ = "GRC Lab, Illinois Institute of Technology"
__email__ = "grc@iit.edu"
__license__ = "BSD-3-Clause"

from .installer import install_iowarp
from .config import get_config, set_config
from .test import run_tests

__all__ = [
    "install_iowarp",
    "get_config", 
    "set_config",
    "run_tests",
]