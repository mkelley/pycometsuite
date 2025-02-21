import pytest
import astropy.units as u
from astropy.time import Time
from mskpy import KeplerState
from .. import integrator, particle, rundynamics, generators
from ..simulation import Simulation
from ..integrator.core import Integrator


class Dummy(Integrator):
    def integrate(self, init, dt, beta=0):
        return init


@pytest.fixture(scope="session")
def sim_radius_uniform() -> Simulation:
    """
    {'box': -1,
     'comet': {'kernel': 'None',
               'name': None,
               'r': [149597870.7, 0.0, 0.0],
               'v': [0.0, 30.0, 0.0]},
     'cometsuite': '1.0.1.dev10+g8fb8422.d20241101',
     'date': '2024-11-01 00:00:00.000',
     'header': 1.0,
     'integrator': 'FreeExpansion()',
     'label': 'None',
     'nparticles': 2000,
     'pfunc': {'age': 'Uniform(x0=0, x1=100)',
               'composition': 'Geometric(rho0=1.0)',
               'density_scale': 'UnityScaler()',
               'radius': 'Uniform(x0=1, x1=10)',
               'speed': 'Delta(x0=0)',
               'speed_scale': 'UnityScaler()',
               'vhat': 'Sunward(body_basis=[[1.,0.,0.],\n'
                       ' [0.,1.,0.],\n'
                       " [0.,0.,1.]], w=None, distribution='uniformangle', "
                       'theta_dist=Delta(x0=0), phi_dist=Delta(x0=0))'},
     'save': ['radius',
              'graindensity',
              'beta',
              'age',
              'origin',
              'r_i',
              'v_ej',
              'r_f'],
     'syndynes': False
    }

    (Pdb) sim.particles
    rec.array([(7.29560845, 1., 0.07812919, 8294549.50082234, [ 85.75209525,   0.        ], [-1.12505320e+07, -1.51469143e+08, -0.00000000e+00], [ 0.,  0.,  0.], [ 1.61195921e+08, -1.49628291e+07,  0.00000000e+00]),
            (2.9806057 , 1., 0.1912363 , 8638853.40827032, [ 81.91801337,   0.        ], [-2.13739527e+07, -1.50520445e+08, -0.00000000e+00], [ 0.,  0.,  0.], [ 1.76509803e+08, -4.01358975e+07,  0.00000000e+00]),
            (7.65856891, 1., 0.07442644, 3119526.89824946, [144.22089909,   0.        ], [ 1.21687493e+08, -8.76963307e+07,  0.00000000e+00], [-0.,  0.,  0.], [ 1.51665496e+08, -8.76436654e+05,  0.00000000e+00]),
            ...,
            (7.08331623, 1., 0.08047078, 8556151.68656234, [ 82.83829627,   0.        ], [-1.89493445e+07, -1.50810094e+08, -0.00000000e+00], [ 0.,  0.,  0.], [ 1.61995788e+08, -1.67375978e+07,  0.00000000e+00]),
            (4.21408061, 1., 0.13526082,   94833.80584007, [178.91036531,   0.        ], [ 1.49571198e+08, -2.84484509e+06,  0.00000000e+00], [-0.,  0.,  0.], [ 1.49601485e+08, -4.58222582e+01,  0.00000000e+00]),
            (9.24739617, 1., 0.06163897, 5550120.78603663, [116.57360843,   0.        ], [ 6.74450685e+07, -1.34839787e+08,  0.00000000e+00], [-0.,  0.,  0.], [ 1.54553926e+08, -3.88137427e+06,  0.00000000e+00])],
            dtype=[('radius', '<f8'), ('graindensity', '<f8'), ('beta', '<f8'), ('age', '<f8'), ('origin', '<f8', (2,)), ('r_i', '<f8', (3,)), ('v_ej', '<f8', (3,)), ('r_f', '<f8', (3,))])
    """

    date = Time("2024-11-01")
    comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 30], date)

    pgen = particle.Coma(
        comet,
        date,
    )
    pgen.composition = particle.Geometric()
    pgen.radius = generators.Uniform(1, 10)
    pgen.age = generators.Uniform(0, 100)
    pgen.nparticles = 2000

    sim = rundynamics.run(pgen, integrator.BulirschStoer(), seed=24)
    sim.observer = KeplerState([0, u.au.to("km"), 0], [30, 0, 0], sim.params["date"])
    sim.observe()

    return sim


@pytest.fixture(scope="session")
def sim_radius_log() -> Simulation:
    date = Time("2024-11-01")
    comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 30], date)

    pgen = particle.Coma(
        comet,
        date,
    )
    pgen.composition = particle.Geometric()
    pgen.radius = generators.Log(-1, 1)
    pgen.age = generators.Uniform(0, 100)
    pgen.nparticles = 2000

    sim = rundynamics.run(pgen, integrator.BulirschStoer(), seed=24)
    sim.observer = KeplerState([0, u.au.to("km"), 0], [30, 0, 0], sim.params["date"])
    sim.observe()

    return sim


@pytest.fixture(scope="session")
def sim_radius_log_big() -> Simulation:
    """More particles: takes about 1 min to generate."""
    date = Time("2024-11-01")
    comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 30], date)

    pgen = particle.Coma(
        comet,
        date,
    )
    pgen.composition = particle.Geometric()
    pgen.radius = generators.Log(-1, 4)
    pgen.age = generators.Uniform(0, 100)
    pgen.nparticles = 100000

    sim = rundynamics.run(pgen, Dummy(), seed=25)
    sim.observer = KeplerState([0, u.au.to("km"), 0], [30, 0, 0], sim.params["date"])
    sim.observe()

    return sim
