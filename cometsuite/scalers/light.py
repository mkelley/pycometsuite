__all__ = [
    "LightScaler",
    "ScatteredLight",
    "ThermalEmission",
]

import numpy as np
from scipy.interpolate import splrep, splev
import astropy.units as u
from sbpy.calib import Sun
from mskpy import planck
from .core import Scaler


class MissingGrainModel(Exception):
    pass


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
