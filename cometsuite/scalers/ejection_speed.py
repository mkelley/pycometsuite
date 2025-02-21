__all__ = [
    "EjectionSpeedScaler",
    "SpeedLimit",
    "SpeedRadius",
    "SpeedRh",
]

import numpy as np
from .core import Scaler


class EjectionSpeedScaler(Scaler):
    """Abstract base class for scalers that affect the ejection speed."""


class SpeedLimit(EjectionSpeedScaler):
    """Limit speed to given values.

    If the particle speed is outside the range [`smin`, `smax`], the
    returned scale factor is 0.0.  1.0, otherwise.


    Parameters
    ----------
    smin : float, optional
      Minimum ejection speed. [km/s]

    smax : float, optional
      Maximum ejection speed. [km/s]

    scales : Scaler or CompositeScaler, optional
      Normalize the speed with `scales` before applying limits.  For
      example, if a simulation was picked using over a range of
      values, then scaled with `SpeedRadius`, set `scales` to use the
      same SpeedRadius to undo the scaling.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, smin=0, smax=np.inf, scalers=None):
        self.smin = smin
        self.smax = smax
        if scalers is None:
            self.scalers = UnityScaler()
        else:
            self.scalers = scalers

    def __str__(self):
        return "SpeedLimit(smin={}, smax={}, scalers={})".format(
            self.smin, self.smax, self.scalers
        )

    def scale(self, p):
        s = p.s_ej / self.scalers.scale(p)
        i = (s < self.smin) + (s > self.smax)
        if np.iterable(i):
            scale = np.ones_like(s)
            if any(i):
                scale[i] = 0.0
        else:
            scale = 0.0 if i else 1.0
        return scale


class SpeedRadius(EjectionSpeedScaler):
    """Speed scale factor based on grain raidus.

    For `a` measured in micrometers::

      scale = (a / a0)**k


    Parameters
    ----------
    k : float, optional
      Power-law exponent.

    a0 : float, optional
      Normalization radius.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, k=-0.5, a0=1.0):
        self.k = k
        self.a0 = a0

    def __str__(self):
        return "SpeedRadius(k={}, a0={})".format(self.k, self.a0)

    def scale(self, p):
        return (p.radius / self.a0) ** self.k


class SpeedRh(EjectionSpeedScaler):
    """Speed scale factor based on :math:`|r_i|`.

    For `rh` measured in AU::

      scale = (rh / rh0)**k


    Parameters
    ----------
    k : float, optional
        Power-law exponent.

    rh0 : float, optional
        Normalization distance.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, k=-0.5, rh0=1.0):
        self.k = k
        self.rh0 = rh0

    def __str__(self):
        return "SpeedRh(k={}, rh0={})".format(self.k, self.rh0)

    def scale(self, p):
        if len(p) == 1:
            rh = np.sqrt(np.sum(p.r_i**2)) / 149597870.69
        else:
            rh = np.sqrt(np.sum(p.r_i**2, 1)) / 149597870.69
        return (rh / self.rh0) ** self.k
