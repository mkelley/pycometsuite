import pytest
import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.time import Time
from mskpy.ephem import KeplerState
from ..particle import Coma, AmorphousCarbon
from .. import generators as gen
from .. import scalers as sc
from ..calibration import mass_calibration
from . import sim_radius_uniform, sim_radius_log, sim_radius_log_big


def get_sim_comet(sim):
    comet = KeplerState(
        sim.params["comet"]["r"],
        sim.params["comet"]["v"],
        sim.params["date"],
    )
    return comet


class TestMassCalibration:
    def create_sim(self, pgen):
        sim = pgen.sim()
        sim.init_particles()
        for i, p in enumerate(pgen):
            p.final = p.init
            sim[i] = p
        return sim

    def test_simple_examples(self):
        date = Time("2024-11-01")
        comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 30], date)
        pgen = Coma(comet, date)
        pgen.age = gen.Uniform(0, 1)
        pgen.radius = gen.Uniform(1, 1)
        pgen.composition = AmorphousCarbon()
        pgen.density_scale = sc.UnityScaler()
        pgen.nparticles = 1

        sim = self.create_sim(pgen)
        scaler = sc.UnityScaler()

        # calibrate to 1 kg/s = 86400 kg / day
        Q0 = 1 * u.kg / u.s
        C, M = mass_calibration(sim, scaler, Q0, state_class=KeplerState)

        # the simulation is 1 day long --> 86400 kg expected
        assert np.isclose(M.to_value("kg"), 86400)

        # one 1 um grain in 1 day:
        #     mass = 4 / 3 * pi * (1 * u.um)**3 * 1.5 * u.g / u.cm**3
        #          = 6.283185307179585e-15 kg
        #     calibration = 86400 / 6.283185307179585e-15 = 1.375098708313976e+19
        assert np.isclose(C, 1.375098708313976e19)

        # three grains, picked uniformly from 0.1 to 1
        #     actual mass = 7.33483345531e-15 kg
        #     mean particle mass for the uniform distribution
        #         = 4 / 3 * np.pi * 1500 * (1e-6**4 - 0.1e-6**4) / 4 / 0.9e-6
        #         = 1.7451547190691294e-15
        #     calibration = 86400 / (1.7451547190691294e-15 * 3)
        #         = 1.650283478324604e+19
        pgen.radius = gen.Grid(0.1, 1, 3)
        pgen.nparticles = 3
        sim = self.create_sim(pgen)
        scaler = sc.UnityScaler()
        C, M = mass_calibration(sim, scaler, Q0, state_class=KeplerState)

        assert np.isclose(M.to_value("kg"), 86400)
        assert np.isclose(C, 1.650283478324604e19)

        # same, but weight by a**-2
        #     let dn/da = a**-2 for a in um
        #         --> normalization, N1 = 1e-6**2 for a in m
        #     mean particle mass for uniform distribution
        #         = 4 / 3 * np.pi * 1500 * 1e-12 * (1e-6**2 - 0.1e-6**2) / 2 / 0.9e-6
        #         = 3.455751918948772e-15
        #     calibration = 86400 / (3.455751918948772e-15 * 3)
        #         = 8.333931565539247e+18
        scaler = sc.PSD_PowerLaw(-2)
        C, M = mass_calibration(sim, scaler, Q0, state_class=KeplerState)

        assert np.isclose(M.to_value("kg"), 86400)
        assert np.isclose(C, 8.333931565539247e18)

    def test_simple_examples_with_log_bias(self):
        date = Time("2024-11-01")
        comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 30], date)
        pgen = Coma(comet, date)
        pgen.age = gen.Uniform(0, 1)
        pgen.radius = gen.Log(0, 0)
        pgen.composition = AmorphousCarbon()
        pgen.density_scale = sc.UnityScaler()
        pgen.nparticles = 1

        sim = self.create_sim(pgen)
        scaler = sc.PSD_RemoveLogBias()

        # calibrate to 1 kg/s = 86400 kg / day
        Q0 = 1 * u.kg / u.s
        C, M = mass_calibration(sim, scaler, Q0, state_class=KeplerState)

        # the simulation is 1 day long --> 86400 kg expected
        assert np.isclose(M.to_value("kg"), 86400)

        # one 1 um grain in 1 day:
        #     mass = 4 / 3 * pi * (1 * u.um)**3 * 1.5 * u.g / u.cm**3
        #          = 6.283185307179585e-15 kg
        #     calibration = 86400 / 6.283185307179585e-15 = 1.375098708313976e+19
        assert np.isclose(C, 1.375098708313976e19)

        # repeat, but with a 10 um grain
        #
        #     mass = m(10)
        #          = 6.283185307179585e-12 kg
        #     calibration = 86400 / 6.283185307179585e-12

        pgen.radius = gen.Log(1, 1)
        sim = self.create_sim(pgen)

        C, M = mass_calibration(sim, scaler, Q0, state_class=KeplerState)
        assert np.isclose(M.to_value("kg"), 86400)
        assert np.isclose(C, 1.375098708313976e16)

        # repeat, but pick 3 grains uniformly in log space:
        #
        #     total particle mass based on the (uniform) distribution
        #         = 4 / 3 * np.pi * 1500 * (10e-6**4 - 1e-6**4) / 4 / 9e-6
        #         = 1.7451547190691308e-12
        #     normalize by actual simulation PSD
        #         ∫ a**-1 da = log(10e-6)- log(1e-6) = 2.302585092994045
        #     calibration = 86400 / (1.7451547190691308e-12 / 2.302585092994045 * 3)

        pgen.nparticles = 3
        pgen.radius = gen.Grid(0, 1, pgen.nparticles, log=True)
        sim = self.create_sim(pgen)

        C, M = mass_calibration(sim, scaler, Q0, state_class=KeplerState)
        assert np.isclose(M.to_value("kg"), 86400)
        assert np.isclose(C, 3.799918136404591e16)

    def simulation_mass(self, sim, scaler):
        a = sim.radius * u.um
        rho = sim.graindensity * u.g / u.cm**3
        rh = sim.rh_i

        _scaler = sc.CompositeScaler(scaler)
        dnda = _scaler.filter((sc.ConstantFactor, sc.PSDScaler))
        # .get(
        #    sc.PSD_RemoveLogBias, inverse=True
        # )
        Q = _scaler.filter(sc.ProductionRateScaler)

        Q_norm = Q.scale_rh(np.linalg.norm(sim.params["comet"]["r"]) / u.au.to("km"))
        M = (
            (
                4
                / 3
                * np.pi
                * rho
                * a**3
                * dnda.scale(sim.particles)
                * Q.scale_rh(rh)
                / Q_norm
            )
            .sum()
            .to("kg")
        )

        # normalize for simulation PSD
        # if len(_scaler.get(sc.PSD_RemoveLogBias)) != 0:
        #     M *= len(a) / a.value.sum()

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
        C, M = mass_calibration(sim_radius_uniform, scaler, Q0, state_class=KeplerState)

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
        C, M = mass_calibration(sim_radius_log, scaler, Q0, state_class=KeplerState)

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
        C, M = mass_calibration(sim_radius_log_big, scaler, Q0, state_class=KeplerState)

        assert np.isclose(M, m_cal)
        assert np.isclose(C, expected, rtol=0.02)
