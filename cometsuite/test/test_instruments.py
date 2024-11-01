import pytest
import astropy.units as u
from astropy.time import Time
from mskpy import KeplerState
from ..simulation import Simulation
from .. import generators, particle, instruments, integrator, rundynamics


@pytest.fixture
def sim():
    date = Time("2024-11-01")
    comet = KeplerState([u.au.to("km"), 0, 0], [0, 30, 0], date)
    pgen = particle.Coma(comet, date)
    pgen.composition = particle.Geometric()
    pgen.nparticles = 2000
    sim = rundynamics.run(pgen, integrator.BulirschStoer())
    sim.observer = KeplerState([0, u.au.to("km"), 0], [30, 0, 0], sim.params["date"])
    sim.observe()
    return sim


class TestPhotometer:
    def test_init(self):
        inst = instruments.Photometer(5 * u.arcsec)
        assert inst.data.shape == ()

    def test_integrate(self, sim):
        inst = instruments.Photometer(5 * u.arcsec)
        inst.integrate(sim)
        assert inst.data == 2000
