"""
Free expansion (no gravity)
"""

from .core import Integrator
from ..state import State


class FreeExpansion(Integrator):
    """Free expansion (no gravity)."""

    def integrate(self, init, dt, beta=0):
        return State(init.r + init.v * dt, init.v, init.t + dt / 86400.0)

    integrate.__doc__ = Integrator.integrate.__doc__
