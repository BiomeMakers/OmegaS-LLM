"""
Omega-S: A Functional Resilience Index for Catastrophic Forgetting
===========================================================
USPTO Patent Pending : Serial No. 
Author: Alberto Acedo

Usage:
    from omega_s import StochasticOmegaS, DistributedOmegaS
"""

from .omega_s import StochasticOmegaS
from .omega_s_distributed import DistributedOmegaS

__version__ = "0.1.0"
__author__ = "Alberto Acedo"
__license__ = "AGPL-3.0 (research); Commercial license required for production use."

__all__ = ["StochasticOmegaS", "DistributedOmegaS"]
