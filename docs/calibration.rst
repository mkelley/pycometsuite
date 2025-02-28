Calibrating simulations
=======================

Simulations can be calibrated to a given mass or mass production rate using the functions `mass_calibration` and `production_rate_calibration`, respectively, with some restrictions.  Ages must be picked from a uniform distribution, and grain radii picked from uniform or log distributions.  The derived calibration factors are based on a simulation's particle functions and the desired simulation scaling functions (e.g., `QRh` or `PSD_PowerLaw`).

The following examples are based on a small simulation of a comet at 2 au:

.. code::

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from mskpy import KeplerState
    >>> import cometsuite as cs
    >>> import cometsuite.generators as gen
    >>> import cometsuite.scalers as sc
    >>> 
    >>> # comet C/2009 P1 (Garradd)
    >>> date = Time("2011-09-11")
    >>> rc = [1.99512556e8, -1.86957354e8, 1.49657485e8]
    >>> vc = [-23.28613574, 10.86086487, 13.85289671]
    >>> comet = KeplerState(rc, vc, date, name="comet")
    >>> 
    >>> pgen = cs.Coma(comet, date)
    >>> pgen.composition = cs.Geometric(rho0=1)
    >>> pgen.age = gen.Uniform(0, 10)
    >>> pgen.radius = gen.Log(-1, 3)
    >>> pgen.nparticles = 2000
    >>> 
    >>> integrator = cs.BulirschStoer()
    >>> sim = cs.run(pgen, integrator, seed=26)
    [run] Expecting 2000 particles
    [simulation] Initialized particle array
    [run] 1000 integrated, ... s/particle, complete at ...
    [run] 2000 integrated, ... s/particle, complete at ...
    [run] 2000 particle states integrated
    [run] Overall, ... seconds per particle


Calibrate to a total mass
-------------------------

Here we want to calibrate the above simulation to a total mass of :math:`10^6` kg.  We will use a power-law differential particle size distribution of the form :math:`a^{-3.5}`, where :math:`a` is the particle radius.  The simulation radii were picked from a logarithmic distribution, so we remove this bias using the `PSD_RemoveLogBias` scaler.

.. code::

    >>> psd = sc.PSD_PowerLaw(-3.5) * sc.PSD_RemoveLogBias()
    >>> M0 = 1e6 * u.kg
    >>> C = cs.mass_calibration(sim, psd, M0)
    >>> C  # doctest: +FLOAT_CMP
    1.7558680838473886e+16

Use this multiplicative factor whenever you want absolutely calibrated data, such as a simulated image of scattered light, or the total cross-sectional area:

    >>> G = C * psd.scale(sim) * sim.cs * u.cm**2
    >>> G.sum().to("km2")
    <Quantity 74.69802588 km2>

In this case, we can verify that the calibration is working as intended.  `sim.m` returns the mass of the particles in g.  Scale this by the particle size distribution scalers and the derived scale factor.  The total mass should be near :math:`10^6` kg:

    >>> m_sim = C * psd.scale(sim) * sim.m * u.g
    >>> m_sim.sum().to("kg")  # doctest: +FLOAT_CMP
    <Quantity 1017976.38652193 kg>


Calibrate to a specific number of particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibration is based on the number of particles originally generated in the simulation.  In some cases, we may want to use a different number.  For example, this simulation may be one of a series of simulations that will be added together to produce a total mass of 1 kg/s.  Or, we may only be analyzing a subset of the simulation.  In such cases, the number of particles may be provided with the `n=` keyword argument:

    >>> sim2 = sim[:1000]  # just analyze the first 1000 particles simulated
    >>> sim2.nparticles
    2000  # number of particles originally simulated, not the number of particles in this object
    >>> len(sim2)
    1000  # number of particles in this object, not the number of particles originally simulated
    >>> C2 = cs.mass_calibration(sim2, psd, M0, n=1000)
    >>> C2  # doctest: +FLOAT_CMP
    3.5117361676947772e+16

Note that this is 2 times larger than `C`.


Calibrate to a production rate
------------------------------

Now, calibrate our previous simulation to a mass production rate of 1 kg/s at the time of the observation using `production_rate_calibration`.  This time, we need to specify the state class that will be used to calculate the heliocentric distance of the object.  This class is not saved in the simulation.  By default, `production_rate_calibration` attempts to find the object with a SPICE kernel, but this behavior can be overridden with the `state_class=` keyword argument:

.. code::

    >>> psd = sc.PSD_PowerLaw(-3.5) * sc.PSD_RemoveLogBias()
    >>> Q0 = 1 * u.kg / u.s
    >>> C, M = cs.production_rate_calibration(sim, psd, Q0, state_class=KeplerState)
    >>> C  # doctest: +FLOAT_CMP
    1.5170700244441438e+16

The total mass produced is also returned:

    >>> M  # doctest: +FLOAT_CMP
    <Quantity 864000. kg>

Repeat the exercise, but with a dust production rate that varies with :math:`r_h^{-6}`:

    >>> Q = sc.QRh(-6)
    >>> scaler = psd * Q
    >>> C2, M2 = cs.production_rate_calibration(sim, scaler, Q0, state_class=KeplerState)
    >>> C2  # doctest: +FLOAT_CMP
    1.2412454528133778e+18
    >>> M2  # doctest: +FLOAT_CMP
    <Quantity 766193.14411619 kg>

Note that 13% less mass was produced.


Calibrate to a specific time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `t0=` keyword argument may be used to calibrate the production rate to other times.  Here, we calibrate to 1 kg/s at the time of perihelion of comet Garradd:

    >>> Tp = Time("2011-12-23")
    >>> C3, M3 = cs.production_rate_calibration(sim, scaler, Q0, t0=Tp, state_class=KeplerState)
    >>> C3  # doctest: +FLOAT_CMP
    2.1074854218594627e+17
    >>> M3  # doctest: +FLOAT_CMP
    <Quantity 130090.37639522 kg>