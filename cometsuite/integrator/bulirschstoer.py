"""
Integration with the Bulirsch-Stoer method.
"""

__all__ = ['BulirschStoer']

import numpy as np
from .core import Integrator
from .bsint import bsint
from ..state import State


class BulirschStoer(Integrator):
    """State integration with the Bulirsch-Stoer method.

    Uses the Bulirsch-Stoer method of Bader and Deuflhard, as
    implemented in the GNU Scientific Library.

    Notes
    -----

    G. Bader and P. Deuflhard, “A Semi-Implicit Mid-Point Rule for
    Stiff Systems of Ordinary Differential Equations.”, Numer. Math.
    41, 373–398, 1983.

    """

    def __init__(self):
        pass

    def __str__(self):
        return "BulirschStoer()"

    def integrate(self, init, dt, beta=0):
        """Integrate the state vector.

        Parameters
        ----------
        init : State
            Initial conditions.

        dt : float
            Time to integrate. [s]

        beta : float, optional
            Radiation pressure parameter.

        Returns
        -------
        final : State

        """

        rv = bsint(init.rv, init.t, dt, beta)
        return State(rv[:3], rv[3:], init.t + dt / 86400)
