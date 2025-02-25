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
        # kg
        _scaler = sc.CompositeScaler(scaler).filter(
            (sc.PSDScaler, sc.ProductionRateScaler)
        )
        M = (
            4
            / 3
            * np.pi
            * (1e3 * sim.particles.graindensity)
            * (1e-6 * sim.particles.radius) ** 3
            * _scaler.scale(sim)
        ).sum()
        return M

    @pytest.mark.parametrize(
        "scaler,expected_C,expected_M",
        [
            (sc.UnityScaler(), 3713137826230356.5, 8_640_000),
            (sc.PSD_Constant(10), 371313782623035.65, 8_640_000),
            (sc.PSD_PowerLaw(-3), 1.0313240312354817e18, 8_640_000),
            (sc.PSD_PowerLaw(-4), 4.0310850223780823e18, 8_640_000),
            (sc.QRh(-4), 3713137826230356.5, 8_640_000 * 0.4354466),
            (
                sc.PSD_PowerLaw(-3) * sc.QRh(-4),
                1.0313240312354817e18,
                8_640_000 * 0.4354466,
            ),
            (sc.ScatteredLight(0.6), 3713137826230356.5, 8_640_000),
        ],
    )
    def test_mass_calibration_radius_uniform(
        self, sim_radius_uniform, scaler, expected_C, expected_M
    ):
        """
        Calculate the expected calibration constant:

        Q0 = 1 kg/s
        dt = 100 days
        --> M = 8_640_000 kg

        N = 2000 particles
        a = 1 to 10 μm
        rho = 1.0 g/cm3

        Integrate a in units of m, t in units of s.

                  N                        ∫ dm/dt dt
        M_sim = ----- ∫ m(a) dn/da w(a) da ----------
                ∫ da                         ∫ dt

        (1) dn/da = UnityScaler = 1
            w(a) = 1
            dm/dt = 1

                    4 pi rho N   (a_max**4 - a_min**4)
            M_sim = ---------- * ---------------------
                      3 * 4         (a_max - a_min)

                  = 2.326872958758841e-09

            C = 8640000 / 2.326872958758841e-09
              = 3713137826230356.5

        (2) dn/da = 10

            M_sim = 10 M_sim(1)

            C = 371313782623035.65

        (3) dn/da = PSD_PowerLaw(-3)
                  = a**-3 for a in um
                  = (1e-6 a)**3 for a in m

                       4 pi rho N
            M_sim = ----------------- ∫ a**3 (1e-6 a)**-3 da
                    3 (a_max - a_min)

                    4 pi rho N
            M_sim = ---------- 1e-18
                        3

                  = 8.377580409572782e-12

            C = 1.0313240312354817e+18

        (4) dn/da = a**-4

                       4 pi rho N
            M_sim = ----------------- ∫ a**3 (1e-6 a)**-4 da
                    3 (a_max - a_min)

                    4 pi rho N       (ln(a_max) - ln(a_min))
            M_sim = ---------- 1e-24 -----------------------
                        3                (a_max - a_min)

                  = 2.143343529604581e-12

            C = 4.0310850223780823e+18

        (5) dn/da = 1, Q = rh**-4

                           ∫ dm/dt dt
            --> C = C(1) * ----------
                              ∫ dt

            date = Time("2024-11-01")
            comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 30], date)

            >>> rh = lambda t: np.linalg.norm(comet.r(date - t * u.s))
            >>> rh0 = rh(0)
            >>> quad(lambda t: (rh(t) / rh0)**-4, 0, 8_640_000)[0]  # 1 kg/s
            3762258.9902648977

            3762258.9902648977 / 8640000 = 0.43544664239177056

            M = M(1) * 0.43544664239177056
            M_sim = M_sim(1) * 0.43544664239177056
            --> C = C(1)

        (6) sc.PSD_PowerLaw(-3) * sc.QRh(-4)
            M = M(4)
            C = C(3)

        (7) ScatteredLight(0.6)
            M = M(1)
            C = C(1)

        """

        Q0 = 1 * u.kg / u.s
        C, M = mass_calibration(sim_radius_uniform, scaler, Q0, state_class=KeplerState)
        m_sim = C * self.simulation_mass(sim_radius_uniform, scaler)

        assert np.isclose(M, expected_M * u.kg)
        assert np.isclose(C, expected_C)
        assert np.isclose(m_sim, expected_M, rtol=0.1)

        # expected = (m_cal / m_sim).to_value("")
        # C, M = mass_calibration(sim_radius_uniform, scaler, Q0, state_class=KeplerState)

        # # For 2000 particles of _uniform_ mass, expect an uncertainty of sqrt(2000)
        # # / 2000 = 2.2%
        # assert np.isclose(M, m_cal)
        # assert np.isclose(C, expected, rtol=0.03)

    @pytest.mark.parametrize(
        "scaler,expected",
        [
            (sc.PSD_RemoveLogBias(), 1),
            (sc.PSD_RemoveLogBias() * sc.ConstantFactor(10), 1),
            (sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3), 1),
            (sc.PSD_RemoveLogBias() * sc.QRh(-4), 1),
            (sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3) * sc.QRh(-4), 1),
        ],
    )
    def test_mass_calibration_radius_log(self, sim_radius_log, scaler, expected):
        """
        Calculate the expected calibration constant:

                ∫∫ m(a) dn/da w(a) dm/dt da dt
        M_sim = ------------------------------
                          ∫ dt



        """
        pass
        # m_sim = self.simulation_mass(sim_radius_log, scaler)
        # Q0 = 1 * u.kg / u.s
        # m_cal = self.calibrated_mass(sim_radius_log, scaler, Q0)

        # expected = (m_cal / m_sim).to_value("")
        # C, M = mass_calibration(sim_radius_log, scaler, Q0, state_class=KeplerState)

        # assert np.isclose(M, m_cal)
        # assert np.isclose(C, expected, rtol=0.03)

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
