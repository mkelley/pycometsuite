__all__ = [
    "MassScaler",
    "FractalPorosity",
]

from .core import Scaler


class MassScaler(Scaler):
    """Abstract base class for scalers that affect particle mass."""


class FractalPorosity(MassScaler):
    """Density scale factor based on fractal porosity.

    For the bulk material density `rho0`, minimum grain size `a0`, and
    fractal dimension `D`::

        rho = rho0 * (a / a0)**(D - 3)


    Parameters
    ----------
    D : float
        Fractal dimension.

    a0 : float, optional
        Minimum grian size.  Particles smaller than this will always be
        solid. [micrometer]


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, D, a0=0.1):
        self.D = D
        self.a0 = a0

    def __str__(self):
        return "FractalPorosity(D={}, a0={})".format(self.D, self.a0)

    def formula(self):
        return r"$P = (a / a_0)^{{D-{:.3f}}}$".format(self.a0, self.D)

    def scale(self, p):
        return (p.radius / self.a0) ** (self.D - 3.0)
