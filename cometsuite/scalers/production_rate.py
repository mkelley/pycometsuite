__all__ = [
    "ProductionRateScaler",
    "QRh",
    "QRhDouble",
]

import abc
from .core import Scaler


class ProductionRateScaler(Scaler):
    """Abstract base class for production rate scale factors."""

    @abc.abstractmethod
    def scale_rh(self, rh):
        raise NotImplemented


class QRh(ProductionRateScaler):
    """Dust production rate dependence on `rh_i`.

    :math:`Qd \propto rh_i**k`


    Parameters
    ----------
    k : float
      Power-law scale factor slope.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return "QRh({})".format(self.k)

    def formula(self):
        return (r"$Q \propto r_h^{{{}}}$").format(self.k)

    def scale(self, p):
        return self.scale_rh(p.rh_i)

    def scale_rh(self, rh):
        return rh**self.k


class QRhDouble(ProductionRateScaler):
    """Double power-law version of `QRh`.

    :math:`Qd \propto rh_i**k1` for :math:`rh_i < rh0`
    :math:`Qd \propto rh_i**k2` for :math:`rh_i > rh0`

    The width of the transition from `k1` to `k2` is parameterized by
    `k12`.  Larger `k12` yields shorter transitions.  Try 100.

    The function is normalized to 1.0 at `rh0`.


    Parameters
    ----------
    k1, k2 : float
      Power-law scale factor slopes.

    k12 : float
      Parameter controlling the width of the transition from `k1` to
      `k2`.

    rh0 : float
      The transition heliocentric distance. [AU]


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, k1, k2, k12, rh0):
        self.k1 = k1
        self.k2 = k2
        self.k12 = k12
        self.rh0 = rh0

    def __str__(self):
        return "QRhDouble({}, {}, {}, {})".format(self.k1, self.k2, self.k12, self.rh0)

    def formula(self):
        return (
            r"""$Q \propto r_h^{{{}}}$ for $r_h < {}$ AU
$Q \propto r_h^{{{}}}$ for $r_h > {}$ AU"""
        ).format(self.k1, self.rh0, self.k2, self.rh0)

    def scale(self, p):
        return self.scale_rh(p.rh_i)

    def scale_rh(self, rh):
        alpha = (self.k1 - self.k2) / self.k12
        sc = 2**-alpha
        sc = sc * (rh / self.rh0) ** self.k2
        sc = sc * (1 + (rh / self.rh0) ** self.k12) ** alpha
        return sc
