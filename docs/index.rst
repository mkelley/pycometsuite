
cometsuite
==========

Cometary dust dynamics simulator.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Quick start
-----------

Syndynes
________

Reproduce the syndynes of Reach et al. (2000) for comet 2P/Encke

.. plot::
    :include-source:

    >>> import numpy as np
    >>> from astropy.time import Time
    >>> import cometsuite as cs
    >>> from mskpy import KeplerState, SolarSysObject
    >>>
    >>> # observation date, comet's position and velocity in km and km/s
    >>> date = Time(2450643.5417, format="jd")
    >>> r = [4.36955224e7, -1.64073549e8, -2.73990869e7]
    >>> v = [26.40398803, -21.02309023, -1.63325062]
    >>>
    >>> # the comet object produces state vectors based on Keplerian (two-body) propagation
    >>> comet = KeplerState(r, v, date)
    >>>
    >>> # the composition defines a relationship between β and size
    >>> # syndynes must use a geometric relationship (beta = 0.57 / a / rho)
    >>> composition = cs.Geometric()
    >>>
    >>> # generates particles for this comet, to be observed at this date
    >>> particle_generator = cs.Coma(comet, date, composition=composition)
    >>>
    >>> # generate syndynes for each of these β values, a 200 day length, and 101 time steps
    >>> beta = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.1]
    >>> ndays = 200
    >>> steps = 101
    >>> cs.syndynes(particle_generator, beta=beta, ndays=ndays, steps=steps)
    >>>
    >>> integrator = cs.Kepler()
    >>> sim = cs.run(particle_generator, integrator)
    >>>
    >>> # to plot the results, we need to observe the particles
    >>> earth = SolarSysObject(
    ...     KeplerState([5.61856527e7, -1.41307139e8, -1.23261993e3],
    ...                 [2.71884498e1, 1.09043893e1, -6.22859821e-04],
    ...                 date)
    ... )
    >>> sim.observer = earth
    >>> sim.observe()
    >>>
    >>> # setup axes and plot in polar coordinates; rotate so that 0 is up
    >>> ax = plt.subplot(polar=True, theta_offset=np.pi / 2)
    >>> cs.synplot(sim, ax=ax)
    >>> ax.set_ylim(0, 2 * 3600)  # 4 degree FOV


Monte Carlo Coma
________________





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
