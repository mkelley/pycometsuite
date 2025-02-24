__all__ = ["mass_calibration"]

import warnings
import numpy as np
from numpy import pi
from scipy.integrate import quad
import astropy.units as u
from mskpy import getspiceobj, cal2time, KeplerState
from . import scalers as sc
from .scalers import *  # needed for eval
from .generators import *  # needed for eval
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
        The simulation to calibrate.  Must be simulated with uniform dust
        production rates and radii picked from uniform or log distributions.

    scaler : Scaler or CompositeScaler
        The simulation scale factors, including ``PDS_RemoveLogBias`` when the
        simulation radii are picked from a log distribution.

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
                             ∫∫ m(a) dn/da dn/da|sim dm/dt da dt
    mean particle mass, m_p: -----------------------------------
                                      ∫∫ dn/da|sim da dt

    where dn/da is the desired differential particle size distribution, and
    dm/dt is the desired mass loss rate.  dn/da|sim is the simulated
    differential particle size distribution (uniform or log).

    total expected mass, M: n * m_p

    total simulated mass = sum(m(a))

    --> C = total_expected_mass / total_simulated_mass

    """

    if not sim.params["pfunc"]["age"].startswith("Uniform("):
        raise ValueError("Uniform particle age generator required.")

    radius_pfunc = eval(sim.params["pfunc"]["radius"])
    if isinstance(radius_pfunc, csg.Uniform) or (
        isinstance(radius_pfunc, csg.Grid) and not radius_pfunc.log
    ):
        psd_sim = lambda a: 1.0
    elif isinstance(radius_pfunc, csg.Log) or (
        isinstance(radius_pfunc, csg.Grid) and radius_pfunc.log
    ):
        psd_sim = lambda a: a**-1.0
    else:
        raise ValueError("Uniform, Log, or Grid particle radius generator required.")

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
    _scaler = sc.CompositeScaler(scaler).filter(
        (
            sc.ConstantFactor,
            sc.MassScaler,
            sc.ProductionRateScaler,
            sc.PSDScaler,
        )
    )

    unsupported = _scaler.filter(sc.MassScaler)
    if len(unsupported) > 0:
        raise ValueError("MassScaler is not yet supported")

    # Split scalers into production rate and PSD scalers.  It may be arbitrary
    # where we account for constant factors, but here we put them with the PSDs.
    # PSD_RemoveLogBias affects accounting for the number of simulated
    # particles, but not the expected mass, so treat it separately.
    Qd_scaler = _scaler.filter(sc.ProductionRateScaler)

    psd_scaler = _scaler.filter((sc.PSDScaler, sc.ConstantFactor)).filter(
        sc.PSD_RemoveLogBias, inverse=True
    )

    remove_log_bias = _scaler.filter(sc.PSD_RemoveLogBias)
    if len(remove_log_bias) == 0 and radius_pfunc == "Log":
        raise ValueError(
            "radius particle function was Log, but ``scaler`` is missing RemoveLogBias."
        )

    # density
    composition = eval("csp." + sim.params["pfunc"]["composition"])
    rho = eval(sim.params["pfunc"]["density_scale"]) * composition.rho0

    # calculate the total desired mass of the simulation, with PSD and
    # production rate weights
    def mass(a):
        p = csp.Particle(radius=a)
        # a in μm, mass in kg
        m = 4 / 3 * pi * (a * 1e-6) ** 3 * psd_sim(a)
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
        psd_norm = 1 / quad(psd_sim, *arange_sim, points=points)[0]
        m_p = (quad(mass, *arange_sim, points=points)[0] * psd_norm) * u.kg

    gen = eval("csg." + sim.params["pfunc"]["age"])
    trange_sim = np.array((gen.min(), gen.max())) * 86400  # s
    x = quad(relative_production_rate, *trange_sim)[0] / (np.ptp(trange_sim))

    M_sim = n * m_p * x

    # calculate the desired mass

    # normalize to Q0 at t0
    rh0 = np.linalg.norm(comet.r(t0)) / 1.495978707e8
    Q_normalization = Q0.to("kg/s").value / Qd_scaler.scale_rh(rh0)

    M = Q_normalization * quad(relative_production_rate, *trange_sim)[0] * u.kg
    breakpoint()
    return (M / M_sim).to_value(u.dimensionless_unscaled), M
