Analyzing simulations with instruments
======================================

The `Instrument` class may be used to analyze cometsuite simulations.  They can be designed to mimic specific instrument and filter combinations (e.g., LDT's LMI with an r' filter, or Spitzer's MIPS 24 μm imager).  What they observe is very flexible.  They can image a comet as if seen by a real telescope, or they can produce images of, e.g., average particle size, or age on a pixel-by-pixel basis.

The core function of an `Instrument` is to project simulations on to the sky. Two general instrument classes are provided: `cometsuite.instruments.Camera` for imaging simulations in the plane of the sky, and `cometsuite.instruments.Photometer` for analyzing simulations within an aperture.

The following examples use this simple coma simulation:

.. plot::
    :context:
    :nofigs:
    :include-source:
    :show-source-link: False

    >>> import matplotlib.pyplot as plt
    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from mskpy import KeplerState
    >>> import cometsuite as cs
    >>> import cometsuite.generators as gen
    >>> import cometsuite.scalers as sc
    >>> 
    >>> # observation date, position (km) and velocity (km/s) of
    >>> # comet and observer
    >>> date = Time("2011-09-11")
    >>> # comet:
    >>> rc = [1.99512556e8, -1.86957354e8, 1.49657485e8]
    >>> vc = [-23.28613574, 10.86086487, 13.85289671]
    >>> # Earth:
    >>> re = [1.47206597e8, -3.19139879e7, 1.38953505e2]
    >>> ve = [5.82028904e+00,  2.89897439e+01, -1.16199526e-03]
    >>> 
    >>> # two-body orbit propagation
    >>> comet = KeplerState(rc, vc, date)
    >>> earth = KeplerState(re, ve, date)
    >>> 
    >>> # particle generator
    >>> pgen = cs.Coma(comet, date)
    >>> pgen.composition = cs.Geometric(rho0=1)
    >>> pgen.age = gen.Uniform(0, 5)
    >>> pgen.radius = gen.Log(-1, 3)
    >>> pgen.vhat = gen.Isotropic()
    >>> pgen.speed = gen.Delta(0.3)
    >>> pgen.speed_scale = sc.SpeedRh(-0.5) * sc.SpeedRadius(-0.5)
    >>> pgen.nparticles = 2000
    >>> 
    >>> # generate and integrate particle positions
    >>> integrator = cs.BulirschStoer()
    >>> sim = cs.run(pgen, integrator)
    >>>
    >>> # project particle positions onto the sky
    >>> sim.observer = earth
    >>> sim.observe()


Image the total number of particles
-----------------------------------

Create a 60×60 pixel camera with a pixel scale of 1".  Image the total number of particles per pixel.  The `camera.integrate` method adds a simulation to the camera.  It may be used multiple times to combine simulations.

.. plot::
    :context:
    :nofigs:
    :include-source:
    :show-source-link: False

    >>> camera = cs.Camera(shape=(60, 60), scale=(-1, 1))
    >>> camera.integrate(sim)

`camera.data` holds the resulting image:

.. plot::
    :context:
    :include-source:
    :show-source-link: False

    >>> def plot(camera):
    >>>     fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    >>>     ax.imshow(camera.data, cmap="gray_r", extent=[29, -31, -29, 31], origin="lower")
    >>>     fig.colorbar(ax.images[0])
    >>>     plt.setp(ax, xlabel="RA offset (arcsec)", ylabel="Dec offset (arcsec)")
    >>>     plt.tight_layout()
    >>>
    >>> plot(camera)

.. attention::

    The asymmetry in the extent keyword of `imshow` is needed to align the simulation at the origin.  This may be fixed in a future version.


Scaling simulations
-------------------

Image scattered light
^^^^^^^^^^^^^^^^^^^^^

Particle scalers are used to determine the values that the instrument returns.  Set a camera's `scaler` parameter to use them.

Cometsuite contains a simple description for light scattered by particles, `cometsuite.scalers.ScatteredLight`.  It takes a single parameter, the wavelength of light:

.. plot::
    :context: close-figs
    :include-source:
    :show-source-link: False

    >>> scaler = cs.scalers.ScatteredLight(0.63 * u.um)
    >>> camera = cs.Camera(shape=(60, 60), scale=(-1, 1), scaler=scaler)
    >>> camera.integrate(sim)
    >>> plot(camera)  # note the change in the colorbar scale

.. note::

    `ScatteredLight` approximates Rayleigh scattering, and accounts for
    heliocentric and observer-particle distance, but does not account for albedo
    or phase effects.


Combining particle scalers
^^^^^^^^^^^^^^^^^^^^^^^^^^

Particle scalers may be multiplied together to make a `CompositeScaler`.  For example, scale the ScatteredLight scaler by a constant factor of 1e10:

.. code::

    >>> scaler = cs.scalers.ScatteredLight(0.63 * u.um) * cs.scalers.ConstantFactor(1e10)
    >>> scaler
    ScatteredLight(Quantity("0.63 um"), unit=Unit("W / (um m2)")) * ConstantFactor(10000000000.0)


Account for size distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulations can have particles picked from any size distribution, and the above visualizations are picked from a distribution uniform in log-space via the `Log` generator.  To simulate an image using a typical cometary particle size distribution (PSD), such as :math:`dn/da \propto a^{-3.5}`, two scalar weights must be applied.  First, the biases of the logarithmic distribution must be removed.  This can be done with the `PSD_RemoveLogBias` scaler.  Then, account for the true particle size distribution (at the nucleus) with `PSD_PowerLaw`:


.. plot::
    :context: close-figs
    :include-source:
    :show-source-link: False

    >>> camera.scaler = (
    ...     cs.scalers.ScatteredLight(0.63 * u.um)
    ...     * cs.scalers.PSD_RemoveLogBias()
    ...     * cs.scalers.PSD_PowerLaw(-3.5)
    ... )
    >>> camera.reset()  # clear previous integration
    >>> camera.integrate(sim)
    >>> plot(camera)


.. automodapi:: cometsuite.instruments
    :headings: -^
