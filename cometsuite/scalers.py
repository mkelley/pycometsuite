"""
scalers - Scale factors based on particle parameters.
=====================================================

Note: When new scalers are added, those that may be used by
`ParticleGenerator` should also be added to `ParticleGenerator.reset` .


Classes
-------
Scaler
CompositeScaler

UnityScaler
ConstantFactor
ParameterWeight

MassScaler
FractalPorosity

ProductionRateScaler
QRh
QRhDouble

PSDScaler
PSD_Hanner
PSD_PowerLawLargeScaled
PSD_PowerLaw
PSD_BrokenPowerLaw
PSD_RemoveLogBias

LightScaler
ScatteredLight
ThermalEmission

EjectionDirectionScaler
ActiveArea
Angle
SunCone
GaussianActiveArea
SpeedAngle (Deprecated)
SunSpeedAngle (Deprecated)

EjectionSpeedScaler
SpeedLimit
SpeedRadius
SpeedRh


Functions
---------
flux_scaler
mass_calibrate


Exceptions
----------
InvalidScaler
MissingGrainModel

"""

__all__ = [
    "Scaler",
    "CompositeScaler",
    "ActiveArea",
    "Angle",
    "ConstantFactor",
    "FractalPorosity",
    "GaussianActiveArea",
    "ParameterWeight",
    "PSD_Hanner",
    "PSD_BrokenPowerLaw",
    "PSD_PowerLawLargeScaled",
    "PSD_PowerLaw",
    "PSD_RemoveLogBias",
    "QRh",
    "QRhDouble",
    "ScatteredLight",
    "SpeedLimit",
    "SpeedRadius",
    "SpeedRh",
    "SunCone",
    "SunSpeedAngle",
    "ThermalEmission",
    "UnityScaler",
    "flux_scaler",
]

import abc
from copy import copy
from collections import UserList

import numpy as np
from numpy import pi
from scipy.integrate import quad
from scipy.interpolate import splrep, splev

import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
from sbpy.calib import Sun
from mskpy import getspiceobj, cal2time, planck, KeplerState

from . import util
from . import generators as csg
from . import particle
from . import particle as csp


class InvalidScaler(Exception):
    pass


class MissingGrainModel(Exception):
    pass


class Scaler(abc.ABC):
    """Abstract base class for particle scale factors.

    Notes
    -----
    Particle scale factors are multiplicative.

    """

    def __init__(self):
        pass

    def __mul__(self, scale):
        return CompositeScaler(self, scale)

    def __repr__(self):
        return "<" + str(self) + ">"

    @abc.abstractmethod
    def __str__(self):
        return "Scaler()"

    def copy(self):
        return copy(self)

    def formula(self):
        return ""

    @abc.abstractmethod
    def scale(self, p):
        return 1.0


class CompositeScaler(Scaler, UserList):
    """A collection of multiple scale factors.

    To create a `CompositeScaler`::

        total_scale = CompositeScaler(SpeedRh(), SpeedRadius())
        total_scale = SpeedRh() * SpeedRadius()

    To remove the `SpeedRh` scale::

        del total_scale[0]

    A length-one `CompositeScaler` may also be created::

        s = CompositeScaler(SpeedRh())
        s = SpeedRh() * UnityScaler()

    Iterate over the scaler functions::

        for scaler in scalers:
            scaler


    Raises
    ------
    InvalidScaler
        If a scaler is not an instance of `Scaler`, `CompositeScaler`, `float`,
        or `int`.

    """

    def __init__(self, *scales):
        scales = []
        for sc in scales:
            if isinstance(sc, UnityScaler):
                pass
            elif isinstance(sc, CompositeScaler):
                scales.extend([s.copy() for s in sc])
            elif isinstance(sc, Scaler):
                # must test after CompositeScaler
                scales.append(sc.copy())
            elif isinstance(sc, (float, int)):
                scales.append(ConstantFactor(sc))
            else:
                raise InvalidScaler(sc)
        super(UserList).__init__(scales)

    def __mul__(self, scale):
        result = self.copy()
        result *= scale
        return result

    def __imul__(self, scale):
        if isinstance(scale, Scaler):
            self.append(scale)
        elif isinstance(scale, (CompositeScaler, list, tuple)):
            self.extend(scale)
        elif isinstance(scale, (float, int)):
            self.append(ConstantFactor(scale))
        else:
            raise InvalidScaler(scale)

        return self

    def __repr__(self):
        return "CompositeScaler({})".format(", ".join([repr(s) for s in self]))

    def __str__(self):
        if len(self) == 0:
            return str(UnityScaler())
        else:
            return " * ".join([str(s) for s in self])

    def formula(self):
        return [s.formula() for s in self]

    def scale(self, p):
        c = 1.0
        for s in self:
            c = c * s.scale(p)
        return c

    def scale_a(self, a):
        c = 1.0
        for s in self:
            if hasattr(s, "scale_a"):
                c = c * s.scale_a(a)
        return c

    def scale_rh(self, rh):
        c = 1.0
        for s in self:
            if hasattr(s, "scale_rh"):
                c = c * s.scale_rh(rh)
        return c


class UnityScaler(Scaler):
    """Scale factor of 1.0."""

    def __init__(self):
        pass

    def __str__(self):
        return "UnityScaler()"

    def scale(self, p):
        return np.ones_like(p.radius)

    def scale_a(self, a):
        return np.ones_like(a)

    def scale_rh(self, rh):
        return np.ones_like(rh)


class ConstantFactor(Scaler):
    """Constant scale factor."""

    def __init__(self, c):
        self.c = c

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.c)

    def formula(self):
        return r"$C = {:.3g}$".format(self.c)

    def scale(self, p):
        return self.c * np.ones(len(p))


class ParameterWeight(Scaler):
    """Scale value based on a parameter.


    Parameters
    ----------
    key : string
        The particle parameter key that defines the scale factor, e.g., 'age'.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, key):
        self.key = key

    def __str__(self):
        return 'ParameterWeight("{}")'.format(self.key)

    def formula(self):
        return "W = {}".format(self.key)

    def scale(self, p):
        return p[self.key]


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


class LightScaler(Scaler):
    """Abstract base class for scalers that affect the amount of light."""


class ScatteredLight(LightScaler):
    """Radius-based scaler to simulate light scattering.


    .. note::

        Albedo and phase function are not accounted for.


    The scale factor is::

        Qsca * sigma * S / rh**2 / Delta**2

    where `sigma` is the cross-sectional area of the grain, and `S` is the solar
    flux.  The scattering efficiency is::

        Qsca = (2 * pi * a / wave)**4  for a < wave / 2 / pi

        Qsca = 1.0 for a >= wave / 2 / pi


    Parameters
    ----------
    wave : float
        Wavelength of the light. [micrometers]

    unit : astropy Unit
        The flux density units of the scale factor.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, wave, unit=u.Unit("W/(m2 um)")):
        self.unit = unit
        self.wave = wave

        sun = Sun.from_default()
        # a little wavelength averaging to mitigate absorption line issues
        w = np.r_[0.95, 1.0, 1.05] * wave * u.um
        self.S = sun(w, unit=self.unit).value[1]  # at 1 AU

    def __str__(self):
        return "ScatteredLight({}, unit={})".format(self.wave, repr(self.unit))

    def scale(self, p):
        Q = np.ones_like(p.radius)
        k = self.wave / 2 / np.pi
        i = p.radius < k
        if any(i):
            Q[i] = (p.radius[i] / k) ** 4
        sigma = np.pi * (p.radius * 1e-9) ** 2  # km**2
        return Q * sigma * self.S / p.rh_f**2 / p.Delta**2


class ThermalEmission(LightScaler):
    """Radius-based scaler to simulate thermal emission.

    The scale factor is::

      Qem * sigma * B / Delta**2

    where `sigma` is the cross-sectional area of the grain, and `S` is the solar
    flux.  The scattering efficiency is::

      Qem = 2 * pi * a / wave  for a < wave / 2 / pi Qem = 1.0 for a >= wave / 2
      / pi


    Parameters
    ----------
    wave : float
        Wavelength of the light. [micrometers]

    unit : astropy Unit, optional
        The flux density units of the scale factor.

    composition : Composition, optional
        Use this composition, rather than anything specified in the simulation.

    require_grain_model : bool, optional
        If `True`, and a grain temperature model cannot be found, throw an
        exception.  If `False`, use a blackbody temperature as a fail-safe
        model.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(
        self,
        wave,
        unit=u.Unit("W/(m2 um)"),
        composition=None,
        require_grain_model=False,
    ):
        self.unit = unit
        self.wave = wave
        self.composition = composition
        self.require_grain_model = require_grain_model
        print("ThermalEmission is assuming solid grains at the median rh")

    def __str__(self):
        return (
            "ThermalEmission({}, unit={}, composition={}, " "require_grain_model={})"
        ).format(
            self.wave, repr(self.unit), str(self.composition), self.require_grain_model
        )

    def scale(self, p):
        gtm_filename = {
            "amorphouscarbon": "am-carbon.fits",
            "amorphousolivine50": "am-olivine50.fits",
        }

        if self.composition is None:
            composition = p.params["pfunc"]["composition"].split("(")[0]
        else:
            composition = str(self.composition).split("(")[0]
        composition = composition.lower().strip()

        if composition in gtm_filename:
            from dust import readgtm, gtmInterp

            gtm = readgtm(gtm_filename[composition])
            T = np.zeros_like(p.radius)
            rh = np.median(p.rh_f)

            T_rh = np.zeros_like(gtm[2])
            for i in range(len(gtm[2])):
                T_rh[i] = splev(rh, splrep(gtm[3], gtm[0][0, i]))

            T = splev(p.radius, splrep(gtm[2], T_rh))

            # gtmInterp can't handle all the particles at once
            # for i in xrange(len(p.radius)):
            #    T[i] = gtmInterp(gtm, 3.0, p.radius[i], rh[i])

            # T_rh_a = interpolate.interp2d(gtm[3], gtm[2], gtm[0][0],
            #                              kind='cubic')
            # need to run one at a time?
            # for i in xrange(len(p.radius)):
            #    T[i] = T_rh_a(rh, p.radius[i])
        else:
            if self.require_grain_model:
                raise MissingGrainModel

            T = 278.0 / np.sqrt(p.rh_f)

        Q = np.ones_like(p.radius)
        k = self.wave / 2 / np.pi
        i = p.radius < k
        if any(i):
            Q[i] = p.radius[i] / k
        sigma = np.pi * (p.radius * 1e-9) ** 2  # km**2
        B = planck(self.wave, T, unit=self.unit / u.sr)
        return Q * sigma * B / p.Delta**2


class EjectionDirectionScaler(Scaler):
    """Abstract base class for scalers that are based on ejection direction."""


class ActiveArea(EjectionDirectionScaler):
    """Emission from an active area.


    Parameters
    ----------
    w : float
        Cone full opening angle. [deg]

    ll : array
        Ecliptic longitude and latitude of the pole. [deg]

    func : string
        Scale varies with angle following: sin, sin2, cos, cos2


    Methods
    -------
    scale : Scale factor - 0 to 1 inside cone, 0 outside.

    """

    def __init__(self, w, ll, func=None):
        self.w = w
        self.ll = list(ll)
        self.func = func
        if func == "sin":
            self.f = np.sin
        elif func == "sin2":
            self.f = lambda th: np.sin(th) ** 2
        elif func == "cos":
            self.f = np.cos
        elif func == "cos2":
            self.f = lambda th: np.cos(th) ** 2
        else:
            self.f = lambda th: 1

        # active area normal vector
        a = np.radians(self.ll)
        self.normal = np.array(spherical_to_cartesian(1.0, a[1], a[0]))

    def __str__(self):
        return 'ActiveArea({}, {}, func="{}")'.format(self.w, self.ll, self.func)

    def scale(self, p):
        th = util.angle_between(self.normal, p.v_ej)
        return (th <= (self.w / 2.0)).astype(int) * self.f(th)


class ActiveAreaOld(EjectionDirectionScaler):
    """Emission from an active area.

    Broken, may be more than just spherical_rot.


    Parameters
    ----------
    w : float
        Cone full opening angle. [deg]

    ll : array
        Longitude and latitude of the active area. [deg]

    pole : array
        Ecliptic longitude and latitude of the pole. [deg]

    func : string
        Scale varies with angle following: sin, sin2, cos, cos2


    Methods
    -------
    scale : Scale factor - 0 to 1 inside cone, 0 outside.

    """

    def __init__(self, w, ll, pole, func=None):
        self.w = w
        self.ll = list(ll)
        self.pole = list(pole)
        self.func = func
        if func == "sin":
            self.f = np.sin
        elif func == "sin2":
            self.f = lambda th: np.sin(th) ** 2
        elif func == "cos":
            self.f = np.cos
        elif func == "cos2":
            self.f = lambda th: np.cos(th) ** 2
        else:
            self.f = lambda th: 1

        # pole and origin unit vector
        a = np.radians(self.pole)
        self.pole_unit = np.array(spherical_to_cartesian(1.0, a[1], a[0]))

        # active area normal vector
        pi = np.pi
        print(
            "WARNING: spherical rot must be fixed after final 243P sims (tag old version)"
        )
        # need to rotate pole in the same way that 0,pi/2 is rotated to match ll : use vector_rotate
        o = util.spherical_rot(
            np.radians(pole[0]),
            np.radians(pole[1]),
            0,
            pi / 2,
            np.radians(ll[0]),
            np.radians(ll[1]),
        )
        self.normal = util.lb2xyz(*o)

    def __str__(self):
        return 'ActiveAreaOld({}, {}, {}, func="{}")'.format(
            self.w, self.ll, self.pole, self.func
        )

    def scale(self, p):
        if len(p) > 1:
            dot = np.sum(self.normal * p.v_ej, 1) / p.s_ej
        else:
            dot = np.sum(self.normal * p.v_ej) / p.s_ej
        th = np.degrees(np.arccos(dot))
        return (th <= (self.w / 2.0)).astype(int) * self.f(th)


class Angle(EjectionDirectionScaler):
    """Scale by angle from a vector, with optional constant.

    v = scale * func(th) + const


    Parameters
    ----------
    lam, bet : float
       Ecliptic coordinates of the axis to which angles are measured.
       [deg]

    func : string
        sin, sin2, sin4, cos, cos2, cos4, sin(th/2), etc.,
        sin(th<90), sin2(th<90)

    scale : float
        Scale factor.  [km/s]

    const : float, optional
        Constant offset.  [km/s]

    """

    def __init__(self, lam, bet, func, scale, const=0):
        self.lam = lam
        self.bet = bet
        self.normal = util.lb2xyz(np.radians(lam), np.radians(bet))
        self.func = func
        if func == "sin":
            self.f = np.sin
        elif func == "sin2":
            self.f = lambda th: np.sin(th) ** 2
        elif func == "sin4":
            self.f = lambda th: np.sin(th) ** 4
        elif func == "sin(th/2)":
            self.f = lambda th: np.sin(th / 2)
        elif func == "sin2(th/2)":
            self.f = lambda th: np.sin(th / 2) ** 2
        elif func == "sin4(th/2)":
            self.f = lambda th: np.sin(th / 2) ** 4
        elif func == "sin(th<90)":
            self.f = self.sin_th_lt_90
        elif func == "sin2(th<90)":
            self.f = self.sin2_th_lt_90
        elif func == "cos":
            self.f = np.cos
        elif func == "cos2":
            self.f = lambda th: np.cos(th) ** 2
        elif func == "cos4":
            self.f = lambda th: np.cos(th) ** 4
        elif func == "cos(th/2)":
            self.f = lambda th: np.cos(th / 2)
        elif func == "cos2(th/2)":
            self.f = lambda th: np.cos(th / 2) ** 2
        elif func == "cos4(th/2)":
            self.f = lambda th: np.cos(th / 2) ** 4
        else:
            raise ValueError("func must be sin or cos.")
        self.c1 = scale
        self.c0 = const

    def __str__(self):
        return 'Angle({}, {}, "{}", {}, const={})'.format(
            self.lam, self.bet, self.func, self.c1, self.c0
        )

    @staticmethod
    def sin_th_lt_90(th):
        """sin(th) for th < 90 deg, else 1.0"""
        f = np.sin(th)
        f[th > (np.pi / 2)] = 1.0
        return f

    @staticmethod
    def sin2_th_lt_90(th):
        """sin2(th) for th < 90 deg, else 1.0"""
        f = np.sin(th) ** 2
        f[th > (np.pi / 2)] = 1.0
        return f

    def scale(self, p):
        th = util.angle_between(self.normal, p.v_ej)
        scale = self.c1 * self.f(np.radians(th)) + self.c0
        return scale


class SunCone(Scaler):
    """A cone of emission ejected toward the Sun.


    Parameters
    ----------
    w : float
      Cone full opening angle. [deg]


    Methods
    -------
    scale : Scale factor: 1 inside cone, 0 outside.

    """

    def __init__(self, w):
        self.w = w

    def __str__(self):
        return "SunCone({})".format(self.w)

    def scale(self, p):
        th = util.angle_between(-p.r_i, p.v_ej)
        return (th <= (self.w / 2.0)).astype(int)


class SpeedAngle(Angle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise DeprecationWarning("SpeedAngle depricated; use Angle")


class SunSpeedAngle(SpeedAngle):
    """Scale speed by angle from Sun, with optional constant.

    v = scale * func(th) + const


    Parameters
    ----------
    func : string
        sin, sin2, sin4, cos, cos2, cos4

    scale : float
        Scale factor.  [km/s]

    const : float, optional
        Constant offset.  [km/s]

    """

    def __init__(self, func, scale, const=0):
        super().__init__(0, 0, func, scale, const=const)

    def __str__(self):
        return 'SunSpeedAngle("{}", {}, const={})'.format(
            self.func, self.speed_scale, self.const
        )

    def scale(self, p):
        self.r = (-p.r_i.T / util.magnitude(p.r_i)).T
        return super().scale(p)


class GaussianActiveArea(ActiveArea):
    """Emission from an active area with a normal distribution.

    Planetocentric 0 deg longitude is defined using the North Ecliptic
    Pole: ll00 = pole × NEP.  If the pole is parallel to the NEP, then
    the Vernal Equinox is used instead.


    Parameters
    ----------
    w : float
        Cone full opening angle. [deg]

    sig : float
        Sigma of Gaussian function defining the activity. [deg]

    ll : array
        Longitude and latitude of the active area. [deg]

    pole : array
        Ecliptic longitude and latitude of the pole. [deg]


    Methods
    -------
    scale : Scale factor: 0 to 1 inside the cone, 0 outside.

    """

    def __init__(self, w, sig, ll, pole):
        super().__init__(w, ll, pole)
        self.sig = sig

    def __str__(self):
        return "GaussianActiveArea({}, {}, {}, {})".format(
            self.w, self.sig, self.ll, self.pole
        )

    def scale(self, p):
        th = util.angle_between(self.normal, p.v_ej)
        i = th <= (self.w / 2.0)
        scale = np.zeros(i.shape, float)
        scale[i] = util.gaussian(th[i], 0, self.sig)
        return scale


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


def flux_scaler(Qd=0, psd="a^-3.5", thermal=24, scattered=-1, log_bias=True):
    """Weight a comet simulation with commonly used scalers.


    Parameters
    ----------
    Qd : float, optional
        Specify `k` in `QRh(k)`.

    psd : string, optional
        Particle size distribution, one of 'ism', 'a^k', or 'hanner a0 N ap'.

    thermal : float, optional
        Wavelength of the thermal emission.  Set to <= 0 to disable.
        [micrometers]

    scattered : float, optional
        Wavelength of the scattered light.  Set to <= 0 to disable.
        [micrometers]

    log_bias : bool, optional
        If `True`, include `PSD_RemoveLogBias` in the scaler.


    Returns
    -------
    scale : CompositeScaler

    """

    psd = psd.lower().strip()
    if psd == "ism":
        psd_scaler = PSD_PowerLaw(-3.5)
    elif psd[0] == "a":
        psd_scaler = PSD_PowerLaw(float(psd[2:]))
    elif psd.startswith("hanner"):
        a0, N, ap = [float(x) for x in psd.split()[1:]]
        psd_scaler = PSD_Hanner(a0, N, ap=ap)
    else:
        psd_scaler = UnityScaler()

    if log_bias:
        psd_scaler = psd_scaler * PSD_RemoveLogBias()

    if thermal <= 0:
        therm = UnityScaler()
    else:
        therm = ThermalEmission(thermal)

    if scattered <= 0:
        scat = UnityScaler()
    else:
        scat = ScatteredLight(scattered)

    scaler = QRh(Qd) * psd_scaler * therm * scat
    return scaler


def mass_calibration(sim, scaler, Q0, t0=None, n=None, state_class=None):
    """Calibrate a simulation to an instantaneous dust production rate.

    Requires particles generated uniformly over time.

    .. todo::
        Account for `EjectionDirectionScaler`s.


    Parameters
    ----------
    sim : Simulation
        The simulation to calibrate.

    scaler : Scaler or CompositeScaler
        The simulation scale factors.

    Q0 : Quantity
        The dust production rate (mass per time) at ``t0``.

    t0 : Time, optional
        The dust production rate is specified at this time.  If ``None``, then
        the observation time in params is used.

    n : int or float, optional
        Total number of particles of the simulation.  The default is to use the
        number of simulated particles, but this may not always be desired.

    state_class : class, optional
        Use this state class to determine the position of the comet from "r" and
        "v" in ``sim["comet]``.  The default behavior is to use "name" and
        "kernel" in ``sim["comet"]`` SPICE via ``mskpy.ephem.getspiceobj``.


    Returns
    -------
    calib : float
        The multiplicative calibration factor to scale simulation particles into
        coma particles with the given dust production loss rate.

    total_mass : float
        The total expected mass of the simulation, given ``Q0(rh(t0))`` and
        ``scaler``.


    Notes
    -----
                             ∫∫ m(a) dn/da dm/dt da dt
    mean particle mass, m_p: -------------------------
                                      ∫∫ da dt

    where dn/da is the desired differential particle size distribution, and
    dm/dt is the desired mass loss rate.

    total expected mass, M: ∫ dm/dt dt = total expected mass

    --> C = m_p * n / M

    """

    if not sim.params["pfunc"]["age"].startswith("Uniform("):
        raise ValueError("Uniform particle generator required.")

    n = sim.params["nparticles"] if n is None else n

    # Dust production rate should be normalized to Q0(t0)
    t_obs = cal2time(sim.params["date"])
    if t0 is None:
        t0 = t_obs

    # get comet's position
    if state_class is None:
        if sim.params["comet"]["name"] is None:
            raise ValueError(
                "The comet's name is None: state_class is required but not provided."
            )
        if sim.params["comet"]["kernel"] == "None":
            kernel = None
        else:
            kernel = sim.params["comet"]["kernel"]
        comet = getspiceobj(sim.params["comet"]["name"], kernel=kernel)
    else:
        comet = KeplerState(
            sim.params["comet"]["r"], sim.params["comet"]["v"], sim.params["date"]
        )

    # get all scalers that affect simulation production rate
    _scaler = CompositeScaler(
        *(
            s
            for s in CompositeScaler(scaler)
            if not isinstance(
                s, (LightScaler, EjectionSpeedScaler, ParameterWeight, MassScaler)
            )
        )
    )

    unsupported = [
        not isinstance(s, (ProductionRateScaler, PSDScaler, ConstantFactor))
        for s in _scaler
    ]
    if any(unsupported):
        raise ValueError(
            "Only ProductionRateScaler, PSDScaler, and ConstantFactor are supported."
            "  Either fix the code, or perhaps setting `n` will help?"
        )

    # get scalers that affect dust production rate
    Qd_scaler = CompositeScaler(
        *[s for s in _scaler if isinstance(s, ProductionRateScaler)]
    )

    # get particle size distribution scalers and constant factors (it may be
    # arbitrary where we account for constant factors), do not include
    # PDS_RemoveLogBias
    psd_scaler = CompositeScaler(
        *[
            s
            for s in _scaler
            if (
                isinstance(s, (PSDScaler, ConstantFactor))
                and not isinstance(s, PSD_RemoveLogBias)
            )
        ]
    )

    # density
    composition = eval("csp." + sim.params["pfunc"]["composition"])
    rho = eval(sim.params["pfunc"]["density_scale"]) * composition.rho0

    # calculate the total mass of the simulation, with PSD and production rate
    # weights
    def mass(a):
        # a in μm, mass in kg
        m = 4 / 3 * pi * (a * 1e-6) ** 3
        m *= rho.scale(csp.Particle(radius=a)) * 1e3
        dnda = psd_scaler.scale_a(a)
        return m * dnda

    def relative_production_rate(age):
        # kg/s
        rh = np.linalg.norm(comet.r(t_obs - age * u.s)) / 1.495978707e8
        return Qd_scaler.scale_rh(rh)

    gen = eval("csg." + sim.params["pfunc"]["radius"])
    arange_sim = np.array((gen.min(), gen.max()))

    # mean particle mass of the simulation
    if arange_sim.ptp() == 0:
        m_p = np.squeeze(mass(arange_sim[0])) * u.kg
    else:
        points = np.logspace(np.log10(arange_sim[0]), np.log10(arange_sim[1]), 10)
        psd_norm = 1 / arange_sim.ptp()
        m_p = (quad(mass, *arange_sim, points=points)[0] * psd_norm) * u.kg

    gen = eval("csg." + sim.params["pfunc"]["age"])
    trange_sim = np.array((gen.min(), gen.max())) * 86400  # s
    x = quad(relative_production_rate, *trange_sim)[0] / (trange_sim.ptp())

    M_sim = n * m_p * x

    # calculate total expected mass

    # normalize to Q0 at t0
    rh0 = np.linalg.norm(comet.r(t0)) / 1.495978707e8
    Q_normalization = Q0.to("kg/s").value / Qd_scaler.scale_rh(rh0)

    M = Q_normalization * quad(relative_production_rate, *trange_sim)[0] * u.kg

    return (M / M_sim).to_value(u.dimensionless_unscaled), M
