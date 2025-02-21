__all__ = [
    "PSDScaler",
    "PSD_Hanner",
    "PSD_BrokenPowerLaw",
    "PSD_PowerLaw",
    "PSD_PowerLawLargeScaled",
    "PSD_RemoveLogBias",
]

import abc
import numpy as np
from .core import Scaler


class PSDScaler(Scaler):
    """Abstract base class for particle size distribution factors."""

    @abc.abstractmethod
    def scale_a(self, a):
        raise NotImplemented


class PSD_Hanner(PSDScaler):
    """Hanner modified power law particle size distribuion.

    n(a) = Np * (1 - a0 / a)**M * (a0 / a)**N


    Parameters
    ----------
    a0 : float
        Minimum grain radius. [micrometer]

    N : float
        PSD for large grains (`a >> ap`) is `a**-N`.

    M : float, optional
        `ap = a0 * (M + N) / N`.  One of `M` or `ap` must be provided.

    ap : float, optional
        Peak grain radius.  One of `M` or `ap` must be provided. [micrometer]

    Np : float, optional
        Number of grains with radius `ap`.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, a0, N, M=None, ap=None, Np=1):
        self.a0 = a0
        self.N = N
        self.M = M
        self.ap = ap
        self.Np = 1

        if (M is None) and (ap is None):
            raise ValueError("One of M or ap must be provided.")
        elif M is None:
            self.M = (self.ap / self.a0 - 1) * self.N
        else:
            self.ap = self.a0 * (self.M + self.N) / self.N

    def __str__(self):
        return "PSD_Hanner({}, {}, M={}, ap={}, Np={})".format(
            self.a0, self.N, self.M, self.ap, self.Np
        )

    def formula(self):
        return r"dn/da = {Np:.3g} (1 - {a0:.2g} / a)^M ({a0:.2g} / a)^N".format(
            a0=self.a0, N=self.N, M=self.M, Np=self.Np
        )

    def scale(self, p):
        return self.scale_a(p.radius)

    def scale_a(self, a):
        return self.Np * (1 - self.a0 / a) ** self.M * (self.a0 / a) ** self.N


class PSD_BrokenPowerLaw(PSDScaler):
    """Broken power-law particle size distribution.

    n(a) = N1 * a**N for a < a0
         = N1 * a0**(N - M) * a**M for a > a0


    Parameters
    ----------
    N : float
        Power-law slope.

    a0 : float
        Break point.

    M : float
        Large particle power-law slope.

    N1 : float, optional
        Number of 1-micrometer-radius particles.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, N, a0, M, N1=1):
        self.N = N
        self.a0 = a0
        self.M = M
        self.N1 = N1
        self._small_psd = PSD_PowerLaw(N, N1)
        self._large_psd = PSD_PowerLaw(M, N1 * a0 ** (N - M))

    def __str__(self):
        return "PSD_BrokenPowerLaw({}, {}, {}, N1={})".format(
            self.N, self.a0, self.M, self.N1
        )

    def formula(self):
        return r"$dn/da = {:.3g}\times\,a^{{{:.1f}}}$, $dn/da(a > {:.3g}) = {:.3g}\times\,a^{{{:.1f}}}$".format(
            self.N1, self.N, self.a0, self._large_psd.N1, self.M
        )

    def scale(self, p):
        return self.scale_a(p.radius)

    def scale_a(self, a):
        s = np.ones_like(a)
        small = a < self.a0
        if np.size(s) == 1:
            if small:
                s *= self._small_psd.scale_a(a)
            else:
                s *= self._large_psd.scale_a(a)
        else:
            if np.any(small):
                s[small] = self._small_psd.scale_a(a[small])
            if np.any(~small):
                s[~small] *= self._large_psd.scale_a(a[~small])
        return s


class PSD_PowerLaw(PSDScaler):
    """Power law particle size distribution.

    n(a) = N1 * a**N


    Parameters
    ----------
    N : float
        Power-law slope.

    N1 : float, optional
        Number of 1-micrometer-radius particles.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, N, N1=1):
        self.N = N
        self.N1 = N1

    def __str__(self):
        return "PSD_PowerLaw({}, N1={})".format(self.N, self.N1)

    def formula(self):
        return r"$dn/da = {:.3g}\times\,a^{{{:.1f}}}$".format(self.N1, self.N)

    def scale(self, p):
        return self.scale_a(p.radius)

    def scale_a(self, a):
        return self.N1 * a**self.N


class PSD_PowerLawLargeScaled(PSDScaler):
    """Power-law particle size distribution with enhanced large particles.

    n(a) = N1 * a**N


    Parameters
    ----------
    N : float
        Power-law slope.

    a0 : float
        Grains with a > a0 are enhanced.

    scale_factor : float
        Scale factor for a > a0.

    N1 : float, optional
        Number of 1-micrometer-radius particles.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, N, a0, scale_factor, N1=1):
        self.N = N
        self.a0 = a0
        self.scale_factor = scale_factor
        self.N1 = N1

    def __str__(self):
        return "PSD_PowerLawLargeScaled({}, {}, {}, N1={})".format(
            self.N, self.a0, self.scale_factor, self.N1
        )

    def formula(self):
        return r"$dn/da = {:.3g}\times\,a^{{{:.1f}}}$, $dn/da(a > {:.3g}) = dn/da \times {:.3g}$".format(
            self.N1, self.a0, self.scale_factor, self.N
        )

    def scale(self, p):
        return self.scale_a(p.radius)

    def scale_a(self, a):
        s = self.N1 * a**self.N
        i = a > self.a0
        if np.any(i):
            if np.size(s) == 1:
                s *= self.scale_factor
            else:
                s[i] *= self.scale_factor
        return s


class PSD_RemoveLogBias(PSDScaler):
    """Remove the log bias of a simulation.

    For simulations with radius picked from the Log() generator.


    Parameters
    ----------
    Nt : float, optional
        Normalize to this total number of particles.

    aminmax : array, optional
        Normalize over this radius range.


    Methods
    -------
    scale : Scale factor.

    """

    _Nt = None
    _aminmax = None

    def __init__(self, Nt=None, aminmax=None):
        self.Nt = Nt
        self.aminmax = aminmax

    def __str__(self):
        return "PSD_RemoveLogBias(Nt={}, aminmax={})".format(self.Nt, self.aminmax)

    def formula(self):
        return r"dn/da_{{correction}} = {:.3g} a".format(self.N0)

    @property
    def Nt(self):
        return self._Nt

    @Nt.setter
    def Nt(self, n):
        self._Nt = n
        self._update_N0()

    @property
    def aminmax(self):
        return self._aminmax

    @aminmax.setter
    def aminmax(self, amm):
        self._aminmax = amm
        self._update_N0()

    def _update_N0(self):
        if (self.Nt is not None) and (self.aminmax is not None):
            self.N0 = self.Nt / np.log(max(self.aminmax) / min(self.aminmax))
        else:
            self.N0 = 1.0

    def scale(self, p):
        return self.scale_a(p.radius)

    def scale_a(self, a):
        return self.N0 * a
