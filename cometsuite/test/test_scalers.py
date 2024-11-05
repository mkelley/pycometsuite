import pytest
import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.time import Time
from mskpy.ephem import KeplerState
from .. import generators as gen
from .. import scalers as sc
from ..particle import Particle
from . import sim_no_gravity as sim


@pytest.mark.parametrize(
    "scaler",
    [
        sc.UnityScaler(),
        sc.PSD_PowerLaw(-3),
        sc.QRh(-4),
        sc.PSD_PowerLaw(-3) * sc.QRh(-4),
    ],
)
def test_mass_calibration(scaler, sim):
    a = sim.particles["radius"] * u.um
    rho = sim.particles["graindensity"] * u.g / u.cm**3
    dnda = getattr(scaler, "scale_a", lambda x: 1)
    rh = np.linalg.norm(sim.particles["r_i"], axis=1) / u.au.to("km")
    Q = getattr(scaler, "scale_rh", lambda x: 1)
    M = (4 / 3 * np.pi * rho * a**3 * dnda(a.value) * Q(rh)).sum().to("kg")

    Q0 = 1 * u.kg / u.s
    ages = eval("gen." + sim.params["pfunc"]["age"])
    dt = (ages.x1 - ages.x0) * u.day
    t0 = Time(sim.params["date"])
    comet = KeplerState(
        sim.params["comet"]["r"], sim.params["comet"]["v"], sim.params["date"]
    )

    def dmdt(age):
        rh = np.linalg.norm(comet.r(t0 - age * u.s)) / 1.495978707e8
        return Q(rh)

    M0 = (Q0 * quad(dmdt, ages.x0 * 86400, ages.x1 * 86400)[0] * u.s).to("kg")

    expected = (M0 / M).to_value("")

    C = sc.mass_calibration(sim, scaler, Q0, state_class=KeplerState)
    # For 2000 particles of _uniform_ mass, expect an uncertainty of sqrt(2000)
    # / 2000 = 2.2%
    assert np.isclose(C, expected, rtol=0.03)
