__all__ = [
    "EjectionDirectionScaler",
    "ActiveArea",
    "Angle",
    "SunCone",
    "GaussianActiveArea",
    "SpeedAngle",  # deprecated
    "SunSpeedAngle",  # deprecated
]

import numpy as np
from astropy.coordinates import spherical_to_cartesian
from .core import Scaler
from .. import util


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


class SunCone(EjectionDirectionScaler):
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
