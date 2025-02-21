__all__ = ["mass_calibration"]

import numpy as np
from numpy import pi
from scipy.integrate import quad
import astropy.units as u
from mskpy import getspiceobj, cal2time, KeplerState
from .scalers import *  # needed for eval
from . import scalers as sc
from . import generators as csg  # needed for eval
from . import particle as csp  # needed for eval


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
    _scaler = sc.CompositeScaler(
        *(
            s
            for s in sc.CompositeScaler(scaler)
            if not isinstance(
                s,
                (
                    sc.LightScaler,
                    sc.EjectionSpeedScaler,
                    sc.ParameterWeight,
                    sc.MassScaler,
                ),
            )
        )
    )

    unsupported = [
        not isinstance(s, (sc.ProductionRateScaler, sc.PSDScaler, sc.ConstantFactor))
        for s in _scaler
    ]
    if any(unsupported):
        raise ValueError(
            "Only ProductionRateScaler, PSDScaler, and ConstantFactor are supported."
            "  Either fix the code, or perhaps setting `n` will help?"
        )

    # get scalers that affect dust production rate
    Qd_scaler = sc.CompositeScaler(
        *[s for s in _scaler if isinstance(s, sc.ProductionRateScaler)]
    )

    # get particle size distribution scalers and constant factors (it may be
    # arbitrary where we account for constant factors), do not include
    # PDS_RemoveLogBias
    psd_scaler = sc.CompositeScaler(
        *[
            s
            for s in _scaler
            if (
                isinstance(s, (sc.PSDScaler, sc.ConstantFactor))
                and not isinstance(s, sc.PSD_RemoveLogBias)
            )
        ]
    )

    # density
    composition = eval("csp." + sim.params["pfunc"]["composition"])
    rho = eval(sim.params["pfunc"]["density_scale"]) * composition.rho0

    # calculate the total mass of the simulation, with PSD and production rate
    # weights
    def mass(a):
        p = csp.Particle(radius=a)
        # a in μm, mass in kg
        m = 4 / 3 * pi * (a * 1e-6) ** 3
        m *= rho.scale(p) * 1e3
        dnda = psd_scaler.scale(p)
        return m * dnda

    def relative_production_rate(age):
        # kg/s
        rh = np.linalg.norm(comet.r(t_obs - age * u.s)) / 1.495978707e8
        return Qd_scaler.scale_rh(rh)

    gen = eval("csg." + sim.params["pfunc"]["radius"])
    arange_sim = np.array((gen.min(), gen.max()))

    # mean particle mass of the simulation
    if np.ptp(arange_sim) == 0:
        m_p = np.squeeze(mass(arange_sim[0])) * u.kg
    else:
        points = np.logspace(np.log10(arange_sim[0]), np.log10(arange_sim[1]), 10)
        psd_norm = 1 / np.ptp(arange_sim)
        m_p = (quad(mass, *arange_sim, points=points)[0] * psd_norm) * u.kg

    gen = eval("csg." + sim.params["pfunc"]["age"])
    trange_sim = np.array((gen.min(), gen.max())) * 86400  # s
    x = quad(relative_production_rate, *trange_sim)[0] / (np.ptp(trange_sim))

    M_sim = n * m_p * x

    # calculate total expected mass

    # normalize to Q0 at t0
    rh0 = np.linalg.norm(comet.r(t0)) / 1.495978707e8
    Q_normalization = Q0.to("kg/s").value / Qd_scaler.scale_rh(rh0)

    M = Q_normalization * quad(relative_production_rate, *trange_sim)[0] * u.kg

    return (M / M_sim).to_value(u.dimensionless_unscaled), M
