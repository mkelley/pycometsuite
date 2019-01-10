"""
scalers - Scale factors based on particle parameters.
=====================================================

Note: When new scalers are added, those that may be used by
`ParticleGenerator`s should also be added to `ParticleGenerator.reset`.


Classes
-------
Scaler
CompositeScaler
ProductionRateScaler
PSDScaler

ActiveArea
ConstantFactor
FractalPorosity
GaussianActiveArea
PSD_Hanner
PSD_PowerLaw
PSD_RemoveLogBias
QRh
QRhDouble
ScatteredLight
SpeedLimit
SpeedRadius
SpeedRh
SunCone
ThermalEmission
UnityScaler


Exceptions
----------
InvalidScaler
MissingGrainModel


Functions
---------
flux_scaler
mass_calibrate

"""

__all__ = [
    'Scaler',
    'CompositeScaler',
    'ActiveArea',
    'ConstantFactor',
    'FractalPorosity',
    'GaussianActiveArea',
    'ParameterWeight',
    'PSD_Hanner',
    'PSD_PowerLaw',
    'PSD_RemoveLogBias',
    'QRh',
    'QRhDouble',
    'ScatteredLight',
    'SpeedLimit',
    'SpeedRadius',
    'SpeedRh',
    'SunCone',
    'ThermalEmission',
    'UnityScaler',
    'flux_scaler'
]

import numpy as np
import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
from . import util


class InvalidScaler(Exception):
    pass


class MissingGrainModel(Exception):
    pass


class Scaler(object):
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
        return str(self)

    def __str__(self):
        return '<Scaler>'

    def copy(self):
        return eval(str(self))

    def formula(self):
        return ""

    def scale(self, p):
        return 1.0


class CompositeScaler(Scaler):
    """A collection of multiple scale factors.

    To create a `CompositeScaler`:

      total_scale = CompositeScaler(SpeedRh(), SpeedRadius())
      total_scale = SpeedRh() * SpeedRadius()

    To remove the `SpeedRh` scale::

      del total_scale.scales[0]

    Length-one `CompositeScaler`s may also be created:

      s = CompositeScaler(SpeedRh())
      s = SpeedRh() * UnityScaler()

    Raises
    ------
    InvalidScaler

    """

    def __init__(self, *scales):
        self.scales = []
        for sc in scales:
            if isinstance(sc, UnityScaler):
                pass
            elif isinstance(sc, Scaler):
                self.scales.append(sc.copy())
            elif isinstance(sc, CompositeScaler):
                self.scales.extend([s.copy() for s in sc])
            elif isinstance(sc, (float, int)):
                self.scales.append(ConstantFactor(sc))
            else:
                raise InvalidScaler(sc)

    def __mul__(self, scale):
        result = self.copy()
        result *= scale
        return result

    def __imul__(self, scale):
        if isinstance(scale, Scaler):
            self.scales.append(scale)
        elif isinstance(scale, CompositeScaler):
            self.scales.extend(scale.scales)
        elif isinstance(scale, (float, int)):
            self.scales.append(ConstantFactor(scale))
        else:
            raise InvalidScaler(scale)

        return self

    def __str__(self):
        if len(self.scales) == 0:
            return str(UnityScaler())
        else:
            return ' * '.join([str(s) for s in self.scales])

    def formula(self):
        return [s.formula() for s in self.scales]

    def scale(self, p):
        c = 1.0
        for s in self.scales:
            c = c * s.scale(p)
        return c

    def scale_a(self, a):
        c = 1.0
        for s in self.scales:
            if hasattr(s, 'scale_a'):
                c = c * s.scale_a(a)
        return c

    def scale_rh(self, rh):
        c = 1.0
        for s in self.scales:
            if hasattr(s, 'scale_rh'):
                c = c * s.scale_rh(rh)
        return c


class ProductionRateScaler(Scaler):
    """Abstract base class for production rate scale factors."""

    def scale_rh(self, rh):
        raise NotImplemented


class PSDScaler(Scaler):
    """Abstract base class for particle size distribution factors."""

    def scale_a(self, a):
        raise NotImplemented


class ActiveArea(Scaler):
    """Emission from an active area.


    Parameters
    ----------
    w : float
        Cone full opening angle. [deg]

    ll : array
        Longitude and latitude of the active area. [deg]

    pole : array
        Ecliptic longitude and latitude of the pole. [deg]


    Methods
    -------
    scale : Scale factor - 1 inside cone, 0 outside.

    """

    def __init__(self, w, ll, pole):
        self.w = w
        self.ll = list(ll)
        self.pole = list(pole)

        # pole and origin unit vector
        a = np.radians(self.pole)
        self.pole_unit = np.array(
            spherical_to_cartesian(1.0, a[1], a[0]))

        # active area normal vector
        pi = np.pi
        o = util.spherical_rot(np.radians(pole[0]), np.radians(pole[1]),
                               0, pi / 2,
                               np.radians(ll[0]), np.radians(ll[1]))
        self.normal = util.lb2xyz(*o)

    def __str__(self):
        return 'ActiveArea({}, {}, {})'.format(self.w, self.ll, self.pole)

    def scale(self, p):
        if len(p) > 1:
            dot = np.sum(self.normal * p.v_ej, 1) / p.s_ej
        else:
            dot = np.sum(self.normal * p.v_ej) / p.s_ej
        th = np.degrees(np.arccos(dot))
        return (th <= (self.w / 2.0)).astype(int)


class ConstantFactor(Scaler):
    """Constant scale factor."""

    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'ConstantFactor({})'.format(self.c)

    def formula(self):
        return r"$C = {:.3g}$".format(self.c)

    def scale(self, p):
        return self.c * np.ones(len(p))


class FractalPorosity(Scaler):
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
        return 'FractalPorosity(D={}, a0={})'.format(self.D, self.a0)

    def formula(self):
        return r"$P = (a / a_0)^{{D-{:.3f}}}$".format(self.a0, self.D)

    def scale(self, p):
        return (p.radius / self.a0)**(self.D - 3.0)


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
        return 'GaussianActiveArea({}, {}, {}, {})'.format(
            self.w, self.sig, self.ll, self.pole)

    def scale(self, p):
        if len(p) > 1:
            dot = np.sum(self.normal * p.v_ej, 1) / p.s_ej
        else:
            dot = np.sum(self.normal * p.v_ej) / p.s_ej
        th = np.degrees(np.arccos(dot))
        i = th <= (self.w / 2.0)
        scale = np.zeros(i.shape, float)
        scale[i] = util.gaussian(th[i], 0, self.sig)
        return scale


class ParameterWeight(Scaler):
    """Scale value based on a parameter.

    Parameters
    ----------
    key : string
      The particle parameter key that defines the scale factor, e.g.,
      'age'.

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
      Peak grain radius.  One of `M` or `ap` must be
      provided. [micrometer]
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
            raise ValueError('One of M or ap must be provided.')
        elif M is None:
            self.M = (self.ap / self.a0 - 1) * self.N
        else:
            self.ap = self.a0 * (self.M + self.N) / self.N

    def __str__(self):
        return 'PSD_Hanner({}, {}, M={}, ap={}, Np={})'.format(
            self.a0, self.N, self.M, self.ap, self.Np)

    def formula(self):
        return r"dn/da = {Np:.3g} (1 - {a0:.2g} / a)^M ({a0:.2g} / a)^N".format(
            a0=self.a0, N=self.N, M=self.M, Np=self.Np)

    def scale(self, p):
        return (self.Np * (1 - self.a0 / p.radius)**self.M
                * (self.a0 / p.radius)**self.N)


class PSD_PowerLaw(PSDScaler):
    """Power law particle size distribuion.

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

    def __init__(self, N, Np=1):
        self.N = N
        self.Np = Np

    def __str__(self):
        return 'PSD_PowerLaw({}, Np={})'.format(self.N, self.Np)

    def formula(self):
        return r"$dn/da = {:.3g}\times\,a^{{{:.1f}}}$".format(
            self.Np, self.N)

    def scale(self, p):
        return self.scale_a(p.radius)

    def scale_a(self, a):
        return self.Np * a**self.N


class PSD_RemoveLogBias(PSDScaler):
    """Remove the log bias of a simulation.

    For simulations with radius picked from the Log() generator.

    Parameters
    ----------
    Nt : float, optional
    aminmax : array, optional
      Normalize to `Nt` total particles over the radius range
      `aminmax`.

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
        return 'PSD_RemoveLogBias(Nt={}, aminmax={})'.format(
            self.Nt, self.aminmax)

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


class QRh(ProductionRateScaler):
    """Dust production rate dependence on `rh_i`.

    Qd \propto rh_i**k

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
        return 'QRh({})'.format(self.k)

    def formula(self):
        return (r"$Q \propto r_h^{{{}}}$").format(self.k)

    def scale(self, p):
        return self.scale_rh(p.rh_i)

    def scale_rh(self, rh):
        return rh**self.k


class QRhDouble(ProductionRateScaler):
    """Double power-law version of `QRh`.

    Qd \propto rh_i**k1 for rh_i < rh0
    Qd \propto rh_i**k2 for rh_i > rh0

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
        return 'QRhDouble({}, {}, {}, {})'.format(self.k1, self.k2,
                                                  self.k12, self.rh0)

    def formula(self):
        return (r"""$Q \propto r_h^{{{}}}$ for $r_h < {}$ AU
$Q \propto r_h^{{{}}}$ for $r_h > {}$ AU""").format(self.k1, self.rh0,
                                                    self.k2, self.rh0)

    def scale(self, p):
        return self.scale_rh(p.rh_i)

    def scale_rh(self, rh):
        alpha = ((self.k1 - self.k2) / self.k12)
        sc = 2**-alpha
        sc = sc * (rh / self.rh0)**self.k2
        sc = sc * (1 + (rh / self.rh0)**self.k12)**alpha
        return sc


class ScatteredLight(Scaler):
    """Radius-based scaler to simulate light scattering.

    The scale factor is::

      Qsca * sigma * S / rh / Delta**2

    where `sigma` is the cross-sectional area of the grain, and `S` is
    the solar flux.  The scattering efficiency is::

      Qsca = (2 * pi * a / wave)**4  for a < wave / 2 / pi
      Qsca = 1.0                     for a >= wave / 2 / pi

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

    def __init__(self, wave, unit=u.Unit('W/(m2 um)')):
        self.unit = unit
        self.wave = wave

    def __str__(self):
        return 'ScatteredLight({}, unit={})'.format(self.wave, repr(self.unit))

    def scale(self, p):
        from mskpy.calib import solar_flux
        Q = np.ones_like(p.radius)
        k = self.wave / 2 / np.pi
        i = p.radius < k
        if any(i):
            Q[i] = (p.radius[i] / k)**4
        sigma = np.pi * (p.radius * 1e-9)**2  # km**2
        S = solar_flux(self.wave, unit=self.unit).value  # at 1 AU
        return Q * sigma * S / p.rh_f**2 / p.Delta**2


class SpeedLimit(Scaler):
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
        return 'SpeedLimit(smin={}, smax={}, scalers={})'.format(
            self.smin, self.smax, self.scalers)

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


class SpeedRadius(Scaler):
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
        return 'SpeedRadius(k={}, a0={})'.format(self.k, self.a0)

    def scale(self, p):
        return (p.radius / self.a0)**self.k


class SpeedRh(Scaler):
    """Speed scale factor based on |r_i|.

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
        return 'SpeedRh(k={}, rh0={})'.format(self.k, self.rh0)

    def scale(self, p):
        if len(p) == 1:
            rh = np.sqrt(np.sum(p.r_i**2)) / 149597870.69
        else:
            rh = np.sqrt(np.sum(p.r_i**2, 1)) / 149597870.69
        return (rh / self.rh0)**self.k


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
        return 'SunCone({})'.format(self.w)

    def scale(self, p):
        if len(p) > 1:
            dot = np.sum(-p.r_i * p.v_ej, 1) / p.d_i / p.s_ej
        else:
            dot = np.sum(-p.r_i * p.v_ej) / p.d_i / p.s_ej
        th = np.degrees(np.arccos(dot))
        return (th <= (self.w / 2.0)).astype(int)


class ThermalEmission(Scaler):
    """Radius-based scaler to simulate thermal emission.

    The scale factor is::

      Qem * sigma * B / Delta**2

    where `sigma` is the cross-sectional area of the grain, and `S` is
    the solar flux.  The scattering efficiency is::

      Qem = 2 * pi * a / wave  for a < wave / 2 / pi
      Qem = 1.0                for a >= wave / 2 / pi

    Parameters
    ----------
    wave : float
      Wavelength of the light. [micrometers]
    unit : astropy Unit, optional
      The flux density units of the scale factor.
    composition : Composition, optional
      Use this composition, rather than anything specified in the
      simluation.
    require_grain_model : bool, optional
      If `True`, and a grain temperature model cannot be found, throw
      an exception.  If `False`, use a blackbody temperature as a
      fail-safe model.

    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, wave, unit=u.Unit('W/(m2 um)'),
                 composition=None, require_grain_model=False):
        self.unit = unit
        self.wave = wave
        self.composition = composition
        self.require_grain_model = require_grain_model
        print('ThermalEmission is assuming solid grains at the median rh')

    def __str__(self):
        return (('ThermalEmission({}, unit={}, composition={}, '
                 'require_grain_model={})'
                 ).format(self.wave, repr(self.unit), str(self.composition),
                          self.require_grain_model))

    def scale(self, p):
        from mskpy.util import planck
        from . import particle

        gtm_filename = {'amorphouscarbon': 'am-carbon.fits',
                        'amorphousolivine50': 'am-olivine50.fits'}

        if self.composition is None:
            composition = p.params['pfunc']['composition'].split('(')[0]
        else:
            composition = str(self.composition).split('(')[0]
        composition = composition.lower().strip()

        if composition in gtm_filename:
            from dust import readgtm, gtmInterp
            from scipy import interpolate
            from scipy.interpolate import splrep, splev
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

            T = 278. / np.sqrt(p.rh_f)

        Q = np.ones_like(p.radius)
        k = self.wave / 2 / np.pi
        i = p.radius < k
        if any(i):
            Q[i] = p.radius[i] / k
        sigma = np.pi * (p.radius * 1e-9)**2  # km**2
        B = planck(self.wave, T, unit=self.unit / u.sr)
        return Q * sigma * B / p.Delta**2


class UnityScaler(Scaler):
    """Scale factor of 1.0."""

    def __init__(self):
        pass

    def __str__(self):
        return 'UnityScaler()'

    def scale(self, p):
        return np.ones(len(p))


def flux_scaler(Qd=0, psd='a^-3.5', thermal=24, scattered=-1, log_bias=True):
    """Weight a comet simulation with commonly used scalers.

    Parameters
    ----------
    Qd : float, optional
      Specify `k` in `QRh(k)`.
    psd : string, optional
      Particle size distribution, one of 'ism', 'a^k', or
      'hanner a0 N ap'.
    thermal : float, optional
      Wavelength of the thermal emission.  Set to <= 0 to
      disable. [micrometers]
    scattered : float, optional
      Wavelength of the scattered light.  Set to <= 0 to
      disable. [micrometers]
    log_bias : bool, optional
      If `True`, include `PSD_RemoveLogBias` in the scaler.

    Returns
    -------
    scale : CompositeScaler

    """

    psd = psd.lower().strip()
    if psd == 'ism':
        psd_scaler = PSD_PowerLaw(-3.5)
    elif psd[0] == 'a':
        psd_scaler = PSD_PowerLaw(float(psd[2:]))
    elif psd.startswith('hanner'):
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


def mass_calibrate(Q0, scaler, params, n=None):
    """Calibrate a simulation to an instantaneous dust production rate.

    Currently considers `ProductionRateScaler`s and `PSDScaler`s.

    Requires particles generated uniformly over time.

    Parameters
    ----------
    Q0 : Quantity
      The dust production rate (mass per time) at time of observation.
    scaler : Scaler or CompositeScaler
      The simluation scale factors.
    params : dict
      The parameters of the simulation.
    n : int, optional
      The number of particles in the simulation.  The default is to
      use `params['nparticles']`, but this may not always be desired.

    Returns
    -------
    calib : float
      The calibration factor for the simulation to place simulation
      particles in units of coma particles.

    Notes
    -----

    Let: dm/dp = C * dm/dt * dt/dp

    --> ∫dm/dp dp = C * ∫dm/dt dt / (∫dp/dt dt)

    --> C = ∫dm/dp dp * ∫dp/dt dt / (∫dm/dt dt)

    ∫dm/dp dp = mean particle mass
              = ∫dn/da * 4/3π a**3 da = m_p

    ∫dm/dt dt = total expected mass = M

    ∫dp/dt dt = total simulated particles = n

    --> C = m_p * n / M

    """

    from scipy.integrate import quad
    from mskpy import getspiceobj, cal2time
    from . import generators as csg
    from . import particle as csp

    Q0 = Q0.to(u.kg / u.s)

    if n is None:
        n = params['nparticles']

    if not params['pfunc']['age'].startswith('Uniform('):
        raise ValueError('Uniform particle generator required.')

    gen = eval('csg.' + params['pfunc']['age'])
    trange_sim = np.array((gen.min(), gen.max()))

    gen = eval('csg.' + params['pfunc']['radius'])
    arange_sim = np.array((gen.min(), gen.max()))

    # search scaler for production rate scalers
    Q = UnityScaler()
    if isinstance(scaler, ProductionRateScaler):
        Q *= scaler
    elif isinstance(scaler, CompositeScaler):
        s = [sc for sc in scaler.scales
             if isinstance(sc, ProductionRateScaler)]
        Q *= CompositeScaler(*s)

    # search for PSD scalers
    PSD = UnityScaler()
    if isinstance(scaler, PSDScaler):
        PSD *= scaler
    elif isinstance(scaler, CompositeScaler):
        s = [sc for sc in scaler.scales if (isinstance(sc, PSDScaler))]
        PSD *= CompositeScaler(*s)

    # derive density
    comp = eval('csp.' + params['pfunc']['composition'])
    rho = eval(params['pfunc']['density_scale']) * comp.rho0

    if params['comet']['kernel'] == 'None':
        kernel = None
    else:
        kernel = params['comet']['kernel']

    comet = getspiceobj(params['comet']['name'], kernel=kernel)
    t0 = cal2time(params['date'])

    # normalize to Q0 at t0
    r = comet.r(t0)
    rh = np.sqrt(np.dot(r, r)) / 1.495978707e8
    Q *= Q0.to('kg/s').value / Q.scale_rh(rh)

    def mass(a):
        # a in um, mass in kg
        from numpy import pi
        # m = 4/3. * pi * (a * 1e-4)**3
        m = 4 / 3 * pi * a**3 * 1e-12
        m *= rho.scale(csp.Particle(radius=a)) * 1e-3
        dnda = PSD.scale_a(a)
        return m * dnda

    def production_rate(age):
        # unitless
        r = comet.r(t0 - age * u.s)
        rh = np.sqrt(np.dot(r, r)) / 1.495978707e8
        return Q.scale_rh(rh)

    m_p = (quad(mass, *arange_sim)[0]
           / quad(PSD.scale_a, *arange_sim)[0]**-1) * u.kg
    M = quad(production_rate, *trange_sim)[0] * u.kg * u.day / u.s
    return (M / (m_p * n)).to(u.dimensionless_unscaled).value
