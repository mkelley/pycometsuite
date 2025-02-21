"""
scalers - Scale factors based on particle parameters.
=====================================================

Note: When new scalers are added, those that may be used by `ParticleGenerator`
should also be added to `ParticleGenerator.reset` .

"""

from .core import *
from .ejection_direction import *
from .ejection_speed import *
from .light import *
from .mass import *
from .parameter import *
from .production_rate import *
from .psd import *


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
