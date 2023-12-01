import numpy as np
from numpy import degrees

__all__ = ["Simulation"]


class NoObserver(Exception):
    """The observer is not defined."""

    pass


def particle_property(k):
    def getter(self):
        if k in self.particles.dtype.names:
            return self.particles[k]
        else:
            return None

    def setter(self, v):
        self.particles[k] = v

    return property(getter, setter)


class Simulation(object):
    """A set of simulated particles and RunDynamics parameters.

    ``Simulation(filename, [n=])``
    ``Simulation(sim, [observer=])``
    ``Simulation(**keywords)``


    Parameters
    ----------
    filename : string
        An xyz file name from which to read particles.

    n : int, optional
        Limit the number of particles read from ``filename`` to ``n``.

    sim : Simulation
        A simulation to copy.

    observer : SolarSysObject
        An observer used to project particles onto the Celestial Sphere.

    camera : Camera
        Project the particles onto this camera's array.

    version : string, optional
        When reading from a file, assume this version of XYZFile.

    allocate : bool, optional
        When intializing an empty simulation, set to `False` to prevent
        the `particles` record array from being allocated.

    verbose : bool, optional
        When `False`, feedback to the user will generally be suppressed.

    **keywords
        Any RunDynamics parameter, or ``particles``.


    Attributes
    ----------
    parameters : tuple of strings
        A list of all possible RunDynamics parameters.

    allowedData : tuple of strings
        A list of all allowed particle data names.

    units : string
        The particle data units, as kept in the file.  Note that the
        ``age`` attribute will be in days.

    particles : np.recarray
        The particle data.

    observer : SolarSysObject
        The observer used to project particles onto the sky.

    sky_coords : Projection
        The particles projected onto the sky for ``observer``.

    array_coords : np.recarray
        The particles projected onto a camera array.

    The following attributes directly correspond to those found in
    RunDynamics parameter files:

      comet, kernel, jd, xyzfile, labelprefix (corrsponds to label),
      pfunc, tol, planets, planetlookup, closeapproaches, box, ltt,
      save, synbeta (corresponds to beta), ndays, steps, orbit,
      nparticles, units, data

      label and beta were renamed to allow these attributes to be used
      for the saved data.

    Attributes taken from the particle data:

      radius, graindensity, beta, age, origin, v_ej, r_i, v_i,
      t_i, r_f, v_f, t_f, label

    Attributes generated on the fly:

      s_ej - ejection speed [km/s]
      s_i - initial speed [km/s]
      s_f - final speed [km/s]
      d_i - initial heliocentric distance [km]
      d_f - final heliocentric distance [km]
      d - target-particle distance [km]
      rh_i - initial heliocentric distance [AU]
      rh, rh_f - final heliocentric distance [AU]
      r_c - comet heliocentric coordinates [km]
      m - mass [g]
      cs - cross section [cm^2]
      xyz_heder - A header for XYZFile.

    Attributes taken from `sky_coords` (see `Projection` for details):

      lam, bet, dlam, dbet (from offset), theta, rho, phi, Delta

    Attributes taken from `array_coords`: x, y

    """

    allowedData = (
        "radius",
        "graindensity",
        "beta",
        "age",
        "origin",
        "v_ej",
        "r_i",
        "v_i",
        "t_i",
        "r_f",
        "v_f",
        "t_f",
        "label",
    )
    _dtypes = dict(
        radius="<f8",
        graindensity="<f8",
        beta="<f8",
        age="<f8",
        origin="2<f8",
        v_ej="3<f8",
        r_i="3<f8",
        v_i="3<f8",
        t_i="<f8",
        r_f="3<f8",
        v_f="3<f8",
        t_f="3<f8",
        label="16S",
    )

    def __init__(self, *args, **keywords):
        from .xyzfile import XYZFile0, XYZFile1, params_template, xyz_version

        self.params = params_template.copy()
        self.particles = None
        self.observer = keywords.get("observer")
        self.camera = keywords.get("camera")
        version = keywords.get("version")
        self.verbose = keywords.get("verbose", True)

        if len(args) > 0:
            if type(args[0]) is str:
                # initialize with a file name
                filename = args[0]
                if version is None:
                    version = xyz_version(filename)
                if version[0] == "0":
                    XYZFile = XYZFile0
                elif version[0] == "1":
                    XYZFile = XYZFile1
                else:
                    raise NotImplementedError("XYZFile version {}".format(version))

                with XYZFile(filename, "r", verbose=self.verbose) as inf:
                    sim = inf.read_simulation(n=keywords.get("n"))
                    self.params = sim.params
                    self.particles = sim.particles
            elif type(args[0]) is Simulation:
                # copy another simulation
                sim = args[0]
                self.params = sim.params
                self.particles = sim.particles
                self.verbose = sim.verbose
            else:
                raise TypeError(
                    "Optional arguments are file name or " "a simulation instance."
                )
        else:
            # set up parameters from keywords
            for k, v in keywords.items():
                self.params[k] = v
            if keywords.get("allocate", True):
                self.init_particles()
                if self.verbose:
                    print("[simulation] Initialized particle array")
            else:
                if self.verbose:
                    print("[simulation] Particle array not initialized")

        if self.observer is None:
            self.sky_coords = None
        else:
            self.observe()

        if self.camera is None:
            self.array_coords = None
        else:
            import pdb

            pdb.set_trace()
            self.camera.sky2xy(self)

    def __getitem__(self, k):
        if isinstance(k, str):
            if k in self.allowedData:
                if k in self.particles.dtype.names:
                    return self.particles[k]
                else:
                    return None
            else:
                return getattr(self, k)

        sim = Simulation(self)
        if isinstance(k, (slice, np.ndarray, list)):
            sim.particles = self.particles[k]
        else:
            sim.particles = self.particles[k : k + 1]
        if self.sky_coords is not None:
            sim.sky_coords = self.sky_coords[k]
        if self.array_coords is not None:
            sim.array_coords = self.array_coords[k]
        return sim

    def __setitem__(self, k, v):
        if isinstance(k, str):
            return setattr(self, k, v)
        else:
            rec = ()
            for i in self.particles.dtype.names:
                rec += (v[i],)
            self.particles[k] = rec

    def __len__(self):
        return len(self.particles)

    def __repr__(self):
        from .xyzfile import params2header

        return (
            params2header(self.params)
            + """
Actual number of particles: {}
Average log(beta): {}
Average beta: {}
Average radius (micron): {}
Average density (g/cm3): {}
Average age (days): {}
Average rh (AU): {}
""".format(
                len(self),
                None if self.beta is None else np.log10(self.beta).mean(),
                None if self.beta is None else self.beta.mean(),
                None if self.radius is None else self.radius.mean(),
                None if self.graindensity is None else self.graindensity.mean(),
                None if self.age is None else self.age.mean() / 86400.0,
                None if self.rh is None else self.rh.mean(),
            )
        )

    def _xyz2radec(self, ro, rt):
        """Heliocentric ecliptic rectangular coordinates to RA, Dec."""
        rot = rt - ro

        lam = np.arctan2(rot.T[1], rot.T[0])
        bet = np.arctan2(rot.T[2], np.sqrt(rot.T[0] ** 2 + rot.T[1] ** 2))

        # using the mean obliquity of the ecliptic at the J2000.0 epoch
        # eps = 23.439291111 degrees (Astronomical Almanac 2008)
        ceps = 0.91748206207  # cos(eps)
        seps = 0.39777715593  # sin(eps)

        cbet = np.cos(bet)
        sbet = np.sin(bet)
        clam = np.cos(lam)
        slam = np.sin(lam)

        ra = np.arctan2(ceps * cbet * slam - seps * sbet, cbet * clam)
        sdec = seps * cbet * slam + ceps * sbet
        del cbet, sbet, clam, slam

        if np.iterable(sdec):
            sdec[sdec > 1.0] = 1.0
        else:
            if sdec > 1.0:
                sdec = 1.0
        dec = np.arcsin(sdec)
        ra = (ra + 4.0 * np.pi) % (2.0 * np.pi)  # make sure 0 <= ra < 2pi
        return np.c_[ra, dec].T

    def init_particles(self):
        """Initialize the particle record array.

        The length of the array is based on `params['particles']`.
        The columns are from `params['save']`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        dtypes = [(k, self._dtypes[k]) for k in self.params["save"]]
        self.particles = np.recarray(self.params["nparticles"], dtypes)

    def observe(self):
        """Observe the particles with the observer..

        Use to force an update to `sky_coords`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        from mskpy.util import date2time
        from .projection import Projection

        if self.observer is None:
            raise NoObserver("Observer not defined.")
        if self.verbose:
            print(
                (
                    "[simulation] Observing the simulation from {}".format(
                        self.observer.name
                    )
                )
            )

        date = date2time(self.params["date"])

        ro = self.observer.r(date)  # observer position
        rt = self.r_c  # comet position

        particle_radec = self._xyz2radec(ro, self.r_f)
        comet_radec = self._xyz2radec(ro, rt)
        delta = np.sqrt(np.sum((self.r_f - ro) ** 2, 1))

        self.sky_coords = Projection(
            comet_radec, particle_radec, delta, observer=self.observer
        )

    ######################################################################
    # properties from particles
    radius = particle_property("radius")
    graindensity = particle_property("graindensity")
    beta = particle_property("beta")
    age = particle_property("age")
    origin = particle_property("origin")
    v_ej = particle_property("v_ej")
    r_i = particle_property("r_i")
    v_i = particle_property("v_i")
    t_i = particle_property("t_i")
    r_f = particle_property("r_f")
    v_f = particle_property("v_f")
    t_f = particle_property("t_f")
    label = particle_property("label")

    ######################################################################
    # derived properties
    @property
    def s_ej(self):
        """Ejection speed (km/s)."""
        if self.v_ej is None:
            return None
        return np.sqrt(np.sum(self.v_ej**2, 1))

    @property
    def s_i(self):
        """Initial speed (km/s)."""
        if self.v_i is None:
            return None
        return np.sqrt(np.sum(self.v_i**2, 1))

    @property
    def s_f(self):
        """Final speed (km/s)."""
        if self.v_f is None:
            return None
        return np.sqrt(np.sum(self.v_f**2, 1))

    @property
    def d_i(self):
        """Initial Sun-particle distance (km)."""
        if self.r_i is None:
            return None
        return np.sqrt(np.sum(self.r_i**2, 1))

    @property
    def d_f(self):
        """Final Sun-particle distance (km)."""
        if self.r_f is None:
            return None
        return np.sqrt(np.sum(self.r_f**2, 1))

    @property
    def d(self):
        """Target-particle distance (km)."""
        from mskpy import getxyz

        if self.r_f is None:
            return None
        RHt = getxyz(self.comet, date=self.jd, kernel=self.kernel)[0]
        return np.sqrt(np.sum((self.r_f - RHt) ** 2, 1))

    @property
    def rh_i(self):
        """Initial heliocentric distance (AU)."""
        if self.d_i is None:
            return None
        return self.d_i / 149597870.691

    @property
    def rh_f(self):
        """Final heliocentric distance (AU)."""
        if self.d_f is None:
            return None
        return self.d_f / 149597870.691

    @property
    def rh(self):
        """Final heliocentric distance (AU)."""
        return self.rh_f

    @property
    def r_c(self):
        """Comet heliocentric coordintes at time of observation."""
        from mskpy import getxyz

        if "r" in self.params["comet"]:
            return self.params["comet"]["r"]
        else:
            return getxyz(
                self.params["comet"]["name"],
                date=self.params["date"],
                kernel=self.params["comet"]["kernel"],
            )[0]

    @property
    def m(self):
        """Mass (g)."""
        if (self.radius is None) or (self.graindensity is None):
            return None
        return 4 / 3.0 * np.pi * self.graindensity * (self.radius * 1e-4) ** 3

    @property
    def cs(self):
        """Cross section (cm^2)."""
        if self.radius is None:
            return None
        return np.pi * (self.radius * 1e-4) ** 2

    ######################################################################
    # properties from sky_coords
    @property
    def lam(self):
        """Sky coordinate 0: RA/lambda/etc. [degrees]"""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return degrees(self.sky_coords.particles[0])

    @property
    def bet(self):
        """Sky coordinate 1: Dec/beta/etc. [degrees]."""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return degrees(self.sky_coords.particles[1])

    @property
    def dlam(self):
        """Sky coordinate 0 (RA/lambda/etc.) offset from target."""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return self.sky_coords.offset[0]

    @property
    def dbet(self):
        """Sky coordinate 1 (Dec/beta/etc.) offset from target."""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return self.sky_coords.offset[1]

    @property
    def theta(self):
        """Angular distance from target."""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return self.sky_coords.theta

    @property
    def rho(self):
        """Projected distance from target."""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return self.sky_coords.rho

    @property
    def phi(self):
        """Projected position angle from target, CCW from dlam, dbet = (0, 1)."""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return self.sky_coords.phi

    @property
    def Delta(self):
        """Particle-observer distance. [km]"""
        if self.sky_coords is None:
            raise TypeError("Particles not yet observed.")
        return self.sky_coords.Delta

    ######################################################################
    # properties from array_coords
    @property
    def x(self):
        """Array coordinate x. [pixels]"""
        if self.array_coords is None:
            raise TypeError("Particles not yet imaged.")
        return self.array_coords.x

    @property
    def y(self):
        """Array coordinate y. [pixels]"""
        if self.array_coords is None:
            raise TypeError("Particles not yet imaged.")
        return self.array_coords.y
