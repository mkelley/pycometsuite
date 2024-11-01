"""
particle - All things particle.
===============================

Classes
-------
Particle
ParticleGenerator
Coma

Composition
AmorphousCarbon
Geometric


Exceptions
----------
CompositionError


Functions
---------
syndynes

"""

__all__ = ["Particle", "Coma", "AmorphousCarbon", "Geometric", "syndynes"]

import numpy as np
from scipy.interpolate import interp2d


class Particle(object):
    """A particle.

    Parameters
    ----------
    age : float
        Age. [s]

    init : State
        Initial state.

    final : State
        Final state.

    radius : float
        Radius. [micrometer]

    rho0 : float
        Bulk material density. [g/cm3]

    porosity : float
        Porosity (vacuum fraction).

    beta : float
        Radiation pressure parameter.

    v_ej : array
        Ejection velocity with respect to parent object. [km/s]

    origin : array
        The planetocentric longitude and latitude of the ejection point. [deg]

    label : string
        A label.


    Attributes
    ----------
    age : float
    init : State
    final : State
    radius : float
    rho0 : float
    porosity : float
    beta : float
    v_ej : ndarray
    origin : ndarray
    label : string
        See Parameters for descriptions.

    rho : float
    graindesnity : float
        Particle density.

    r_i : ndarray
    v_i : ndarray
    t_i : float
        Initial position and velocity vectors, and time.  Time is with respect
        to the final state. [km, km/s, s]

    r_f : ndarray
    v_f : ndarray
    t_f : float
        Final position and velocity vectors, and time.  Time is with respect to
        the final state, and, therefore, is always 0. [km, km/s, s]

    """

    def __init__(self, **kwargs):
        self.age = kwargs.pop("age", None)
        self.init = kwargs.pop("init", None)
        self.final = kwargs.pop("final", None)
        self.radius = kwargs.pop("radius", None)
        self.rho0 = kwargs.pop("rho0", None)
        self.porosity = kwargs.pop("porosity", None)
        self.beta = kwargs.pop("beta", None)
        self.v_ej = kwargs.pop("v_ej", None)
        self.origin = kwargs.pop("origin", None)
        self.label = kwargs.pop("label", None)

    def __len__(self):
        # always 1
        return 1

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, item):
        return item in [
            "age",
            "init",
            "final",
            "radius",
            "rho0",
            "porosity",
            "beta",
            "v_ej",
            "origin",
            "label",
            "rho",
            "graindensity",
            "r_i",
            "v_i",
            "t_i",
            "r_f",
            "v_f",
            "t_f",
        ]

    def __repr__(self):
        return "<cometsuite Particle>"

    def __str__(self):
        return (
            """Particle(
       age = {} days
      init = {}
     final = {}
    radius = {} um
      rho0 = {} g/cm3
  porosity = {}
      beta = {}
      v_ej = {} km/s
    origin = {} deg
     label = {}
)"""
        ).format(
            self.age / 86400.0,
            ("\n" + " " * 13).join(str(self.init).splitlines()),
            ("\n" + " " * 13).join(str(self.final).splitlines()),
            self.radius,
            self.rho0,
            self.porosity,
            self.beta,
            self.v_ej,
            self.origin,
            self.label,
        )

    @property
    def graindensity(self):
        """Particle density. [g/cm3]"""
        return self.rho

    @property
    def rho(self):
        """Particle density."""
        return (1 - self.porosity) * self.rho0

    @property
    def r_f(self):
        """Final position vector. [km]"""
        return self.final.r

    @property
    def v_f(self):
        """Final velocity vector. [km/s]"""
        return self.final.v

    @property
    def t_f(self):
        """Time at final state with respect to final state. [s]"""
        return 0.0

    @property
    def r_i(self):
        """Initial position vector. [km]"""
        return self.init.r

    @property
    def v_i(self):
        """Initial velocity vector. [km/s]"""
        return self.init.v

    @property
    def t_i(self):
        """Time at initial state with respect to final state. [s]"""
        return -self.age


class ParticleGenerator(object):
    def __iter__(self):
        return self

    def reset(self):
        """Reset particle generators to their initial state."""
        from .generators import (  # noqa: F401
            CosineAngle,
            Delta,
            Grid,
            Log,
            Normal,
        )
        from .generators import Sequence, Uniform, UniformAngle  # noqa: F401
        from .generators import Isotropic, Sunward  # noqa: F401
        from .scalers import (  # noqa: F401
            FractalPorosity,
            SpeedRadius,
            SpeedRh,
            UnityScaler,
        )

        self.ngenerated = 0
        for k in [
            "age",
            "speed",
            "vhat",
            "radius",
            "speed_scale",
            "density_scale",
        ]:
            setattr(self, k, eval(str(getattr(self, k))))


class Coma(ParticleGenerator):
    """Comet dust grain generator.


    Parameters
    ----------
    comet : SolarSysObject
        The parent comet.

    date : various, optional
        The observation date, processed with `mskpy.date2time`.

    age : Generator, optional
    speed : Generator, optional
    vhat : Generator, optional
    radius : Generator, optional
        Dynamical and physical parameter generators.  Default is `Delta(0)`.
        [days, km/s, and micrometers]

    composition : Composition, optional
        The grain's composition.  Default is ``None``.

    speed_scale : Scaler or CompositeScaler, optional
        Ejection speed scaling object.  Speed is the last parameter picked, so
        that the speed scales may be based on any other particle parameter.
        Default scale is 1.0.

    density_scale : function or tuple, optional
        Grain density scaling object.  Default scale is 1.0.

    nparticles : int, optional
        Set to limit the number of particles, or ``None`` for no limit.  Even
        when nparticles is unlimited, the number of particles may still be
        limited by another generator (e.g., a `Sequence` of particle radii).

    params : dict
        Additonal `Simulation` parameters.  For example, syndynes will
        need `syndynes='True'`.

    verbose : bool, optional
        Enable a chatty program.


    Attributes
    ----------
    jd : float
        Julian date of the observation.

    ngenerated : int
        The number of generated particles.


    Methods
    -------
    next:
        Generate a new particle.

    sim:
        A simulation object, which describes this particle generator.


    Raises
    ------
    StopIteration:
        When ``nparticles`` is reached, or another generator stops.

    """

    def __init__(self, comet, date, **kwargs):
        from mskpy.util import date2time
        from . import generators as gen
        from .scalers import UnityScaler

        self.comet = comet
        self.date = date2time(date)
        self.jd = self.date.jd

        self.age = kwargs.pop("age", gen.Delta(0))
        self.speed = kwargs.pop("speed", gen.Delta(0))
        self.vhat = kwargs.pop("vhat", gen.Sunward())
        self.radius = kwargs.pop("radius", gen.Delta(0))
        self.composition = kwargs.pop("composition", None)

        self.speed_scale = kwargs.pop("speed_scale", UnityScaler())
        self.density_scale = kwargs.pop("density_scale", UnityScaler())

        self.nparticles = kwargs.pop("nparticles", 0)
        self.ngenerated = 0

        self.verbose = kwargs.pop("verbose", False)

        self.params = kwargs.pop("params", dict())

    def sim(self):
        """Describe this generator with a `Simulation`.


        Returns
        -------
        sim : Simulation

        """

        from mskpy import ephem
        from .simulation import Simulation

        sim = Simulation()
        for k, v in self.params.items():
            sim.params[k] = v

        r, v = self.comet.rv(self.jd)
        sim.params["comet"]["r"] = [float(x) for x in r]
        sim.params["comet"]["v"] = [float(x) for x in v]
        sim.params["comet"]["name"] = self.comet.name

        if hasattr(self.comet, "state"):
            sim.params["comet"]["spice name"] = self.comet.state.obj
            if isinstance(self.comet.state, ephem.SpiceState):
                sim.params["comet"]["kernel"] = self.comet.state.kernel

        sim.params["date"] = self.date.iso
        sim.params["nparticles"] = self.nparticles

        keys = [
            "age",
            "radius",
            "composition",
            "density_scale",
            "speed",
            "speed_scale",
            "vhat",
        ]
        if sim.params["syndynes"]:
            # age and radius not needed
            keys = keys[2:]
        for k in keys:
            sim.params["pfunc"][k] = str(getattr(self, k))

        return sim

    def __next__(self):
        """Generate a new particle.


        Returns
        -------
        p : Particle

        """

        from .state import State

        p = Particle()

        p.age = next(self.age) * 86400.0
        jd = self.jd - p.age / 86400.0
        r, v = self.comet.rv(jd)
        p.init = State(r, v, jd)

        p.radius = next(self.radius)
        p.rho0 = self.composition.rho0

        p.porosity = 1 - self.density_scale.scale(p)
        p.beta = self.composition.beta(p.radius, p.porosity)

        p.v_ej, p.origin = (x[0] for x in self.vhat.next(p.init))
        p.v_ej = p.v_ej * self.speed_scale.scale(p) * next(self.speed)
        p.init.v += p.v_ej

        self.ngenerated += 1
        if self.nparticles is not None:
            if self.ngenerated > self.nparticles:
                raise StopIteration

        return p


class Composition(object):
    """Abstract base class for CometSuite materials.


    Parameters
    ----------
    name : string
        The name of this composition.

    rho0 : float
        Bulk material density. [g/cm3]

    Qpr : function
        Radiation pressure efficiency, averaged over the solar spectrum, given
        radius in micrometers.


    Methods
    -------
    beta:
        Radiation pressure beta parameter from radius and porosity.

    radius:
        Radius from radiation pressure beta parameter and porosity.

    """

    def __init__(self, name, rho0, Qpr):
        self.name = name
        self.rho0 = rho0
        self.Qpr = Qpr

    def beta(self, radius, porosity):
        """Radiation pressure beta parameter.


        Parameters
        ----------
        radius : float or array
            Particle radius.  [micrometer]

        porosity : float or array
            Particle porosity (vacuum fraction).


        Returns
        -------
        beta : float or ndarray
            Beta.

        """

        numerator = self.Qpr(radius) * 0.57
        denominator = np.array(radius) * (1 - np.array(porosity)) * self.rho0
        if len(denominator) > 1:
            i = denominator <= 0
            numerator[i] = 0
            denominator[i] = 1
        elif denominator == 0:
            return 0
        return numerator / denominator

    def radius(self, beta, porosity):
        """Particle radius.

        The solution is iteratively derived since Qpr depends on radius.


        Parameters
        ----------
        beta : float
            Particle beta.

        porosity : float
            Particle porosity (vacuum fraction).


        Returns
        -------
        radius : float
            Particle radius.

        """
        radius = 0.57 / beta / self.rho0 / (1 - porosity)
        da = 1
        while da > 1e-4:
            a = self.Qpr(radius) * 0.57 / beta / self.rho0 / (1 - porosity)
            da = abs(a - radius) / a
            radius = a
        return radius


class AmorphousCarbon(Composition):
    """Amorphous carbon grains.

    Only supports porosities up to 99%.


    Methods
    -------
    beta:
        Radiation pressure beta parameter from radius and porosity.

    radius:
        Radius from radiation pressure beta parameter and porosity.

    """

    name = "amorphous carbon"
    rho0 = 1.5

    def __init__(self):
        from . import __path__

        data = np.loadtxt(__path__[0] + "/data/amcarbon-qpr.dat")
        self._p = data[0, 1:]
        self._a = data[1:, 0]
        self._qpr = data[1:, 1:]
        self._interp = interp2d(self._p, self._a, self._qpr)

    def __str__(self):
        return "AmorphousCarbon()".format()

    def Qpr(self, radius, porosity):
        """Radiation pressure efficiency.


        Parameters
        ----------
        radius : float or array
            Particle radius.  [micrometer]

        porosity : float or array
            Particle porosity (vacuum fraction).


        Returns
        -------
        qpr : float or ndarray

        """
        return self._interp(porosity, radius)

    def beta(self, radius, porosity):
        """Radiation pressure beta parameter.


        Parameters
        ----------
        radius : float or array
            Particle radius.  [micrometer]

        porosity : float or array
            Particle porosity (vacuum fraction).


        Returns
        -------
        beta : float or ndarray
            Beta.

        """
        return self.Qpr(radius, porosity) * 0.57 / radius / self.rho0 / (1 - porosity)

    def radius(self, beta, porosity):
        """Particle radius.

        The solution is iteratively derived since Qpr depends on radius.


        Parameters
        ----------
        beta : float
          Particle beta.

        porosity : float
          Particle porosity (vacuum fraction).


        Returns
        -------
        radius : float
          Particle radius.

        """
        radius = 0.57 / beta / self.rho0 / (1 - porosity)
        da = 1
        while da > 1e-4:
            a = self.Qpr(radius, porosity) * 0.57 / beta / self.rho0 / (1 - porosity)
            da = abs(a - radius) / a
            radius = a
        return radius


class Geometric(Composition):
    """Material with properties tied to geometric cross section.

    Qpr = 1.0 for all a.


    Parameters
    ----------
    rho0 : float, optional
        The density of the material. [g/cm3]


    Methods
    -------
    beta:
        Radiation pressure beta parameter from radius and porosity.

    radius:
        Radius from radiation pressure beta parameter and porosity.

    """

    def __init__(self, rho0=1.0):
        Composition.__init__(self, "geometric", rho0, lambda a: 1.0)

    def __str__(self):
        return "Geometric(rho0={})".format(self.rho0)


class CompositionError(Exception):
    pass


def syndynes(pgen, beta=[0.001, 0.01, 0.1, 1], ndays=90, steps=31):
    """Syndyne setup.


    Parameters
    ----------
    pgen : ParticleGenerator
        The particle generator to set up.

    beta : array, optional
        The list of particle betas to generate.

    ndays : float or array, optional
        The length of the syndynes.  If an array, specify the start and
        end age. [days]

    steps : int, optional
        The number of steps to take for each syndyne.


    Raises
    ------
    CompositionError

    """

    from .generators import Sequence, Grid

    assert isinstance(pgen, ParticleGenerator)

    if not isinstance(pgen.composition, Geometric):
        raise CompositionError("Syndynes must use a geometric composition.")

    radius = np.zeros(len(beta))
    for i in range(len(beta)):
        radius[i] = pgen.composition.radius(beta[i], 0)

    if np.size(ndays) == 1:
        ndays = [0, ndays]

    # Compute one radius at a time.
    pgen.radius = Sequence(radius, repeat=steps)
    pgen.age = Grid(ndays[0], ndays[1], steps, cycle=len(radius))
    pgen.nparticles = len(beta) * steps

    pgen.params["syndynes"] = True
