"""
GlueBind: A Tripartite Discriminator for Molecular Glue Ternary Complex Prediction

Author: [Team Name]
Date: 2026-03-27
"""

__version__ = "0.1.0"
__author__ = "[Team Name]"

from .models import GlueBindModel
from .data import TernaryDataset

__all__ = ["GlueBindModel", "TernaryDataset"]
