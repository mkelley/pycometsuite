__all__ = ["mass_calibration"]

import warnings
import numpy as np
from numpy import pi
from scipy.integrate import quad
import astropy.units as u
from mskpy import getspiceobj, cal2time, KeplerState
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
                                 ∫∫ m(a) dn/da dm/dt da dt
    total simulated mass, M_sim: -------------------------
                                       ∫∫ u(a) da dt

    where dn/da is the differential particle size distribution, u(a) is the
    distribution from which particle radii are picked, and dm/dt is the mass
    loss rate scaler.

                             Q0 ∫ dm/dt dt
    total expected mass, M = -------------
                               dm/dt|t=t0
    --> C = M / M_sim

    """

    t_gen = eval("csg." + sim.params["pfunc"]["age"])
    if not isinstance(t_gen, csg.Uniform):
        raise ValueError("Uniform particle age generator required.")

    radius_gen = eval("csg." + sim.params["pfunc"]["radius"])
    if isinstance(radius_gen, csg.Uniform) or (
        isinstance(radius_gen, csg.Grid) and not radius_gen.log
    ):
        psd_sim = sc.UnityScaler()
        psd_correction = sc.UnityScaler()
    elif isinstance(radius_gen, csg.Log) or (
        isinstance(radius_gen, csg.Grid) and radius_gen.log
    ):
        psd_sim = sc.PSD_PowerLaw(-1)
        psd_correction = sc.PSD_Constant(1e6)
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

    # Split scalers into production rate and PSD scalers.
    _scaler = sc.CompositeScaler(scaler)

    # It is not clear if ConstantFactor should be used: is it for the radiation?
    # PSD? Q?  The latter two could be addressed here, but not the first.
    if len(_scaler.filter(sc.ConstantFactor)) != 0:
        raise ValueError("Arbitrary constants (ConstantFactor) cannot be used.")

    if len(_scaler.filter(sc.MassScaler)) > 0:
        raise ValueError(
            "MassScaler is not yet supported, but since mass affects dynamics, "
            "you probably didn't want it here anyway."
        )

    # Split scalers into production rate and PSD scalers.

    Qd_scaler = _scaler.filter(sc.ProductionRateScaler)
    psd_scaler = _scaler.filter(sc.PSDScaler).filter(sc.PSD_RemoveLogBias, inverse=True)

    if len(_scaler.filter(sc.PSD_RemoveLogBias)) == 0 and isinstance(
        radius_gen, csg.Log
    ):
        raise ValueError(
            "radius particle function was Log, but ``scaler`` is missing RemoveLogBias."
        )

    # density
    composition = eval("csp." + sim.params["pfunc"]["composition"])
    rho = eval("sc." + sim.params["pfunc"]["density_scale"]) * composition.rho0

    # size distribution range, and integration points
    gen = eval("csg." + sim.params["pfunc"]["radius"])
    arange_sim = 1e-6 * np.array((gen.min(), gen.max()))  # m
    a_points = np.logspace(np.log10(arange_sim[0]), np.log10(arange_sim[1]), 10)

    # Scale to the total number of simulated particles, n
    if np.ptp(arange_sim) == 0:
        sim_scale = n
        psd_correction = sc.UnityScaler()
    else:
        sim_scale = (
            n / quad(lambda a: psd_sim.scale(csp.Particle(radius=a)), *arange_sim)[0]
        )

    # calculate the total desired mass of the simulation, with PSD and
    # production rate weights

    def mass(a):
        """Mass of a single particle, weighted by the normalized PSD.

        a in m, rho in g/cm3, mass in kg

        """
        p = csp.Particle(radius=1e6 * a)
        m = 4 / 3 * pi * a**3 * rho.scale(p) * 1e3
        dnda = psd_scaler.scale(p) * psd_correction.scale(p)
        return m * dnda

    def production_rate(age):
        # kg/s
        rh = np.linalg.norm(comet.r(t_obs - age * u.s)) / 1.49597871e8
        return Q_normalization * Qd_scaler.scale_rh(rh)

    # total desired mass

    # normalization for production_rate: Q0 at t0
    rh0 = np.linalg.norm(comet.r(t0)) / 1.49597871e8
    Q_normalization = (Q0 / Qd_scaler.scale_rh(rh0)).to_value("kg/s")

    trange_sim = np.array((t_gen.min(), t_gen.max())) * 86400  # s
    if np.ptp(trange_sim) == 0:
        raise ValueError("Cannot calibrate simulations with Δage = 0.")

    x_t = quad(production_rate, *trange_sim)[0]
    M = Q_normalization * x_t

    # total simulated mass
    #
    # We required a simulation with particles distributed uniformly in time.
    # This allows us to separate the radius and time integrals.

    if np.ptp(arange_sim) == 0:
        x_a = mass(arange_sim[0])
    else:
        x_a = sim_scale * quad(mass, *arange_sim, points=a_points)[0]

    M_sim = x_a * x_t / np.ptp(trange_sim)

    return M / M_sim, M * u.kg
