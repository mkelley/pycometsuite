__all__ = ["production_rate_calibration", "mass_calibration"]

import warnings
import numpy as np
from numpy import pi
from scipy.integrate import quad
import astropy.units as u
from mskpy import getspiceobj, cal2time, KeplerState
from . import scalers as sc
from . import generators as csg  # needed for eval
from . import particle as csp  # needed for eval


def production_rate_calibration(sim, scaler, Q0, t0=None, n=None, state_class=None):
    """Calibrate a simulation to an instantaneous dust production rate.

    Requires particles generated uniformly over time.


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
        "v" in ``sim["comet"]``.  The default behavior is to use "name" and
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

    Total mass in absence of production rate scaling, but including production
    rate normalization:

              ∫ Q0 dt
        M0 = ----------
             dm/dt|t=t0

    where dm/dt is the dust production rate scaler.

    Use M0 to calibrate the simulation:

        C = M0 / M_sim

    Total mass with PSD and production rate scaling:

         Q0 ∫ dm/dt dt
    M1 = -------------
           dm/dt|t=t0

    """

    t_gen = eval("csg." + sim.params["pfunc"]["age"])
    if not isinstance(t_gen, csg.Uniform):
        raise ValueError("Uniform particle age generator required.")

    # Dust production rate will be normalized to Q0(t0)
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

    # Get production rate scalers.
    _scaler = sc.CompositeScaler(scaler)
    Qd_scaler = _scaler.filter(sc.ProductionRateScaler)

    # calculate the total desired mass of the simulation given dust production
    # rate weights

    # normalization for production_rate: Q0 at t0
    rh0 = np.linalg.norm(comet.r(t0)) / 1.49597871e8
    Q0_normalized = Q0 / Qd_scaler.scale_rh(rh0)

    # get time range from simulation
    trange_sim = np.array((t_gen.min(), t_gen.max())) * 86400  # s
    if np.ptp(trange_sim) == 0:
        raise ValueError("Cannot calibrate simulations with Δage = 0.")

    def production_rate(age):
        # kg/s
        rh = np.linalg.norm(comet.r(t_obs - age * u.s)) / 1.49597871e8
        return float(Qd_scaler.scale_rh(rh))

    x_t = quad(production_rate, *trange_sim)[0] * Q0_normalized.to_value("kg/s")
    M1 = x_t

    M0 = (Q0_normalized * np.ptp(trange_sim) * u.s).to(u.kg)

    return mass_calibration(sim, scaler, M0, n=n), M1 * u.kg


def mass_calibration(sim, scaler, M0, n=None):
    """Calibrate a simulation to a total dust mass.

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

    M0 : Quantity
        The desired total mass.

    n : int or float, optional
        Total number of particles of the simulation.  The default is to use the
        number of simulated particles, but this may not always be desired.


    Returns
    -------
    calib : float
        The multiplicative calibration factor to scale simulation particles into
        coma particles with the given dust production loss rate.


    Notes
    -----
                                 ∫ m(a) dn/da da
    total simulated mass, M_sim: ---------------
                                    ∫ u(a) da

    where dn/da is the differential particle size distribution, u(a) is the
    distribution from which particle radii are picked, and dm/dt is the mass
    loss rate scaler.

    --> C = M0 / M_sim

    """

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

    # Get PSD scalers.
    _scaler = sc.CompositeScaler(scaler)
    psd_scaler = _scaler.filter(sc.PSDScaler).filter(sc.PSD_RemoveLogBias, inverse=True)

    # Use of ConstantFactor is ambiguous: is it for the radiation? PSD? Q?  The
    # latter two could be addressed here, but not the first.
    if len(_scaler.filter(sc.ConstantFactor)) != 0:
        raise ValueError(
            "Arbitrary constants (ConstantFactor) cannot be used.  Consider using another approach."
        )

    if len(_scaler.filter(sc.MassScaler)) > 0:
        raise ValueError(
            "MassScaler is not yet supported, but since mass affects dynamics, "
            "you probably didn't want it here anyway."
        )

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

    def mass(a):
        """Mass of a single particle, weighted by the normalized PSD.

        a in m, rho in g/cm3, mass in kg

        """
        p = csp.Particle(radius=1e6 * a)
        m = 4 / 3 * pi * a**3 * rho.scale(p) * 1e3
        dnda = psd_scaler.scale(p) * psd_correction.scale(p)
        return (m * dnda)[0]

    # Calculate mass and scale to the total number of simulated particles, n
    if np.ptp(arange_sim) == 0:
        psd_correction = sc.UnityScaler()
        M_sim = n * mass(arange_sim[0])
    else:
        M_sim = (
            n
            * quad(mass, *arange_sim, points=a_points)[0]
            / quad(lambda a: float(psd_sim.scale(csp.Particle(radius=a))), *arange_sim)[
                0
            ]
        )

    return u.Quantity(M0, "kg").value / M_sim
