import pytest
import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.time import Time
from mskpy.ephem import KeplerState
from .. import generators as gen
from .. import scalers as sc
from . import sim_radius_uniform, sim_radius_log, sim_radius_log_big


def get_sim_comet(sim):
    comet = KeplerState(
        sim.params["comet"]["r"], sim.params["comet"]["v"], sim.params["date"]
    )
    return comet


class TestMassCalibration:
    def simulation_mass(self, sim, scaler):
        a = sim.radius * u.um
        rho = sim.graindensity * u.g / u.cm**3
        rh = sim.rh_i

        dnda = getattr(scaler, "scale_a", lambda x: 1)
        Q = getattr(scaler, "scale_rh", lambda x: 1)

        Q_norm = Q(np.linalg.norm(sim.params["comet"]["r"]) / u.au.to("km"))
        M = (4 / 3 * np.pi * rho * a**3 * dnda(a.value) * Q(rh) / Q_norm).sum().to("kg")

        # normalize for simulation PSD
        if any(
            [isinstance(s, sc.PSD_RemoveLogBias) for s in sc.CompositeScaler(scaler)]
        ):
            M *= len(a) / a.value.sum()

        return M

    def calibrated_mass(self, sim, scaler, Q0):
        Q = getattr(scaler, "scale_rh", lambda x: 1)
        t0 = Time(sim.params["date"])
        ages = eval("gen." + sim.params["pfunc"]["age"])
        comet = get_sim_comet(sim)

        def dmdt(age):
            rh = np.linalg.norm(comet.r(t0 - age * u.s)) / 1.495978707e8
            return Q(rh)

        M0 = (Q0 * quad(dmdt, ages.x0 * 86400, ages.x1 * 86400)[0] * u.s).to("kg")

        return M0

    @pytest.mark.parametrize(
        "scaler",
        [
            sc.UnityScaler(),
            sc.ConstantFactor(10),
            sc.PSD_PowerLaw(-3),
            sc.QRh(-4),
            sc.PSD_PowerLaw(-3) * sc.QRh(-4),
            sc.PSD_Hanner(0.1, -3.3, ap=1),
            sc.ScatteredLight(0.6),
        ],
    )
    def test_mass_calibration_radius_uniform(self, sim_radius_uniform, scaler):
        m_sim = self.simulation_mass(sim_radius_uniform, scaler)
        Q0 = 1 * u.kg / u.s
        m_cal = self.calibrated_mass(sim_radius_uniform, scaler, Q0)

        expected = (m_cal / m_sim).to_value("")
        C, M = sc.mass_calibration(
            sim_radius_uniform, scaler, Q0, state_class=KeplerState
        )

        # For 2000 particles of _uniform_ mass, expect an uncertainty of sqrt(2000)
        # / 2000 = 2.2%
        assert np.isclose(M, m_cal)
        assert np.isclose(C, expected, rtol=0.03)

    @pytest.mark.parametrize(
        "scaler",
        [
            sc.PSD_RemoveLogBias(),
            sc.PSD_RemoveLogBias() * sc.ConstantFactor(10),
            sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3),
            sc.PSD_RemoveLogBias() * sc.QRh(-4),
            sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3) * sc.QRh(-4),
        ],
    )
    def test_mass_calibration_radius_log(self, sim_radius_log, scaler):
        m_sim = self.simulation_mass(sim_radius_log, scaler)
        Q0 = 1 * u.kg / u.s
        m_cal = self.calibrated_mass(sim_radius_log, scaler, Q0)

        expected = (m_cal / m_sim).to_value("")
        C, M = sc.mass_calibration(sim_radius_log, scaler, Q0, state_class=KeplerState)

        assert np.isclose(M, m_cal)
        assert np.isclose(C, expected, rtol=0.03)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "scaler",
        [
            sc.PSD_RemoveLogBias(),
            sc.PSD_RemoveLogBias() * sc.ConstantFactor(10),
            sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3),
            sc.PSD_RemoveLogBias() * sc.QRh(-4),
            sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3) * sc.QRh(-4),
        ],
    )
    def test_mass_calibration_radius_log_slow(self, sim_radius_log_big, scaler):
        m_sim = self.simulation_mass(sim_radius_log_big, scaler)
        Q0 = 1 * u.kg / u.s
        m_cal = self.calibrated_mass(sim_radius_log_big, scaler, Q0)

        expected = (m_cal / m_sim).to_value("")
        C, M = sc.mass_calibration(
            sim_radius_log_big, scaler, Q0, state_class=KeplerState
        )

        assert np.isclose(M, m_cal)
        assert np.isclose(C, expected, rtol=0.02)
