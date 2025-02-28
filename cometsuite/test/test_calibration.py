import pytest
import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.time import Time
from mskpy.ephem import KeplerState, Earth
from ..particle import Coma, AmorphousCarbon, Geometric
from .. import generators as gen, scalers as sc, integrator, rundynamics
from ..calibration import mass_calibration, production_rate_calibration
from . import sim_radius_uniform, sim_radius_log, sim_radius_log_big


def get_sim_comet(sim):
    comet = KeplerState(
        sim.params["comet"]["r"],
        sim.params["comet"]["v"],
        sim.params["date"],
    )
    return comet


class TestProductionRateCalibration:
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
        C, M = production_rate_calibration(sim, scaler, Q0, state_class=KeplerState)

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
        C, M = production_rate_calibration(sim, scaler, Q0, state_class=KeplerState)

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
        C, M = production_rate_calibration(sim, scaler, Q0, state_class=KeplerState)

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
        m_sim = self.simulation_mass(sim, scaler)

        # calibrate to 1 kg/s = 86400 kg / day
        Q0 = 1 * u.kg / u.s
        C, M = production_rate_calibration(sim, scaler, Q0, state_class=KeplerState)

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

        pgen = Coma(comet, date)
        pgen.age = gen.Uniform(0, 1)
        pgen.radius = gen.Log(1, 1)
        pgen.composition = AmorphousCarbon()
        pgen.density_scale = sc.UnityScaler()
        pgen.nparticles = 1
        sim = self.create_sim(pgen)
        m_sim = self.simulation_mass(sim, scaler)

        C, M = production_rate_calibration(sim, scaler, Q0, state_class=KeplerState)
        assert np.isclose(M.to_value("kg"), 86400)
        assert np.isclose(C, 1.375098708313976e16)

    def simulation_mass(self, sim, scaler):
        _scaler = sc.CompositeScaler(scaler).filter(
            (sc.PSDScaler, sc.ProductionRateScaler)
        )
        M = sim.m * 1e-3 * _scaler.scale(sim)  # kg
        return M.sum()

    @pytest.mark.parametrize(
        "scaler,expected_C,expected_M",
        [
            (sc.UnityScaler(), 3713137826230356.5, 8_640_000),
            (sc.PSD_Constant(10), 371313782623035.65, 8_640_000),
            (sc.PSD_PowerLaw(-3), 1.031324e18, 8_640_000),
            (sc.PSD_PowerLaw(-4), 4.031085e18, 8_640_000),
            (sc.QRh(-4), 5.94102e16, 5_362_296),
            (
                sc.PSD_PowerLaw(-3) * sc.QRh(-4),
                1.6501184e19,
                5_362_296,
            ),
            (sc.ScatteredLight(0.6), 3713137826230356.5, 8_640_000),
        ],
    )
    def test_radius_uniform(self, sim_radius_uniform, scaler, expected_C, expected_M):
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

        (0) dn/da = UnityScaler = 1
            w(a) = 1
            dm/dt = 1

                    4 pi rho N   (a_max**4 - a_min**4)
            M_sim = ---------- * ---------------------
                      3 * 4         (a_max - a_min)

                  = 2.326872958758841e-09

            C = 8640000 / 2.326872958758841e-09
              = 3713137826230356.5

        (1) dn/da = 10

            M_sim = 10 M_sim(0)

            C = 371313782623035.65

        (2) dn/da = PSD_PowerLaw(-3)
                  = a**-3 for a in um
                  = (1e6 a)**-3 for a in m

                       4 pi rho N
            M_sim = ----------------- ∫ a**3 (1e6 a)**-3 da
                    3 (a_max - a_min)

                    4 pi rho N
            M_sim = ---------- 1e-18
                        3

                  = 8.377580409572782e-12

            C = 1.0313240312354817e+18

        (3) dn/da = a**-4

                       4 pi rho N
            M_sim = ----------------- ∫ a**3 (1e6 a)**-4 da
                    3 (a_max - a_min)

                    4 pi rho N       (ln(a_max) - ln(a_min))
            M_sim = ---------- 1e-24 -----------------------
                        3                (a_max - a_min)

                  = 2.143343529604581e-12

            C = 4.0310850223780823e+18

        (4) dn/da = 1, Q = rh**-4

            >>> import numpy as np
            >>> from scipy.integrate import quad
            >>> from astropy.time import Time
            >>> import astropy.units as u
            >>> from mskpy.ephem import KeplerState
            >>> date = Time("2024-11-01")
            >>> comet = KeplerState([2 * u.au.to("km"), 0, 0], [0, 30, 30], date)
            >>> rh = lambda t: np.linalg.norm(comet.r(date - t * u.s)) / 1.49597871e8
            >>> rh0 = rh(0)
            >>> quad(lambda t: (rh(t) / rh0)**-4, 0, 8_640_000)[0]  # 1 kg/s
            5362295.859953757

            5362295.859953757 / 8640000 = 0.6206360949020553

            M = M(0) * 0.6206360949020553
            M_sim = M_sim(0) * 0.6206360949020553 / rh0**-4
                  = M_sim(0) * 0.6206360949020553 / 2**-4
            --> C = C(0) / 2**-4

        (5) sc.PSD_PowerLaw(-3) * sc.QRh(-4)
            M = M(3)
            C = C(2) / 2**-4

        (6) ScatteredLight(0.6)
            M = M(0)
            C = C(0)

        """

        Q0 = 1 * u.kg / u.s
        C, M = production_rate_calibration(
            sim_radius_uniform, scaler, Q0, state_class=KeplerState
        )
        m_sim = C * self.simulation_mass(sim_radius_uniform, scaler)

        assert np.isclose(M, expected_M * u.kg)
        assert np.isclose(C, expected_C)
        assert np.isclose(m_sim, expected_M, rtol=0.1)

    @pytest.mark.parametrize(
        "scaler,expected_C,expected_M",
        [
            (sc.PSD_RemoveLogBias(), 1899769091293167.5, 8_640_000),
            (
                sc.PSD_RemoveLogBias() * sc.PSD_Constant(10),
                189976909129316.78,
                8_640_000,
            ),
            (
                sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3),
                4.79739e17,
                8_640_000,
            ),
            (
                sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-4),
                1.03132e18,
                8_640_000,
            ),
            (
                sc.PSD_RemoveLogBias() * sc.QRh(-4),
                3.03963e16,
                5_362_296,
            ),
            (
                sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3) * sc.QRh(-4),
                7.675824e18,
                5_362_296,
            ),
        ],
    )
    def test_radius_log(self, sim_radius_log, scaler, expected_C, expected_M):
        """
        Calculate the expected calibration constant:

        Q0 = 1 kg/s
        dt = 100 days
        --> M = 8_640_000 kg

        N = 2000 particles
        a = 0.1 to 10 μm
        rho = 1.0 g/cm3

        Integrate a in units of m, t in units of s.

                    N                                ∫ dm/dt dt
        M_sim = ---------- ∫ m(a) dn/da w(a) dlog(a) ----------
                ∫ dlog(a)                               ∫ dt

                        N                                   ∫ dm/dt dt
              = ---------------- ∫ m(a) dn/da w(a) a**-1 da ----------
                log(a_max/a_min)                               ∫ dt

        (0) dn/da = UnityScaler = 1
            w(a) = 1e6 a  # a in m, remove dn/dlog(a) simulation bias
            dm/dt = 1

                        4 pi rho N
            M_sim = ------------------ 1e6 ∫ a**3 da
                    3 log(a_max/a_min)

                        4 pi rho N     (a_max**4 - a_min**4)
            M_sim = ------------------ --------------------- 1e6
                    3 log(a_max/a_min)           4

                  = 4.547921133993593e-09

            C = 8640000 / 4.547921133993593e-09
              = 1899769091293167.5

        (1) dn/da = 10
            M_sim = 10 M_sim(0)
            C = C(0) / 10

        (2) dn/da = PSD_PowerLaw(-3)
                  = a**-3 for a in um
                  = (1e6 a)**-3 for a in m

                        4 pi rho N
            M_sim = ------------------ 1e6 ∫ a**3 (1e6 a)**-3 da
                    3 log(a_max/a_min)

                        4 pi rho N
                  = ------------------ 1e-12 (a_max - a_min)
                    3 log(a_max/a_min)

                  = 1.80097678707123e-11

            C = 8640000 / 1.6372516246102093e-11
              = 4.797396647210801e+17

        (3) dn/da = (1e6 a)**-4 for a in m

                        4 pi rho N
            M_sim = ------------------ 1e6 ∫ a**3 (1e6 a)**-4 da
                    3 log(a_max/a_min)

                        4 pi rho N
                  = ------------------ 1e-18
                             3

                  = 8.377580409572782e-12

            C = 8640000 / 8.377580409572782e-12
              = 1.0313240312354817e+18

        (4) Q = rh**-4

            M = M(0) * 0.6206360949020553 / 2**-4
            M_sim = M_sim(0) * 0.6206360949020553
            --> C = C(0) / 2**-4

        (5) dn/da = a**-4, Q = rh**-3

            --> C = C(2) / 2**-4

        """

        Q0 = 1 * u.kg / u.s
        C, M = production_rate_calibration(
            sim_radius_log, scaler, Q0, state_class=KeplerState
        )
        m_sim = C * self.simulation_mass(sim_radius_log, scaler)

        assert np.isclose(M, expected_M * u.kg)
        assert np.isclose(C, expected_C)
        assert np.isclose(m_sim, expected_M, rtol=0.1)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "scaler,expected_M",
        [
            (sc.PSD_RemoveLogBias(), 8_640_000),
            (sc.PSD_RemoveLogBias() * sc.PSD_Constant(10), 8_640_000),
            (sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3), 8_640_000),
            (sc.PSD_RemoveLogBias() * sc.QRh(-4), 5_362_296),
            (sc.PSD_RemoveLogBias() * sc.PSD_PowerLaw(-3) * sc.QRh(-4), 5_362_296),
        ],
    )
    def test_radius_log_slow(self, sim_radius_log_big, scaler, expected_M):
        Q0 = 1 * u.kg / u.s  # over 100 days
        C, M = production_rate_calibration(
            sim_radius_log_big, scaler, Q0, state_class=KeplerState
        )
        m_sim = C * self.simulation_mass(sim_radius_log_big, scaler)
        assert np.isclose(m_sim, expected_M, rtol=0.05)


class TestMassCalibration:
    def test_wirtanen_20181212(self):
        """Test mass calibration against comet 46P/Wirtanen 2018 Dec 12 outburst.

        Kelley et al. 2021: G=118 km2, M=1.6e6 kg

        Heliocentric orbital elements from Horizons

        Target body name: 46P/Wirtanen                    {source: JPL#K243/4}
        Center body name: Sun (10)                        {source: DE441}
        Center-site name: BODY CENTER

        JDTDB
           X     Y     Z
           VX    VY    VZ
           LT    RG    RR

        2458464.500000000 = A.D. 2018-Dec-12 00:00:00.0000 TDB
         X = 2.282690083831503E-01 Y = 1.030296925323399E+00 Z =-1.779101269594815E-02
         VX=-2.063521603473429E-02 VY= 4.480659466413741E-03 VZ= 4.378202076601094E-03
         LT= 6.095662234191731E-03 RG= 1.055431198445367E+00 RR=-1.628370274983429E-04

        """

        date = Time("2012-12-12")
        comet = KeplerState(
            [2.282690083831503e-01, 1.030296925323399e00, -1.779101269594815e-02]
            * u.au,
            [2.063521603473429e-02, 4.480659466413741e-03, 4.378202076601094e-03]
            * u.au
            / u.day,
            Time("2018-12-12", scale="tdb").utc,
        )

        pgen = Coma(
            comet,
            date,
        )
        pgen.composition = Geometric()
        pgen.radius = gen.Log(-1, 3)
        pgen.age = gen.Uniform(0, 0.1)
        pgen.nparticles = 2000

        sim = rundynamics.run(pgen, integrator.FreeExpansion(), seed=25)
        sim.observer = Earth
        sim.observe()

        psd = sc.PSD_PowerLaw(-3.5) * sc.PSD_RemoveLogBias()

        # calibrate to total mass of 1.6e6 kg
        M0 = 1.6e6 * u.kg
        C = mass_calibration(sim, psd, M0)
        G = (C * psd.scale(sim) * sim.cs).sum() * u.cm**2
        assert np.isclose(G, 118 * u.km**2, rtol=0.05)
