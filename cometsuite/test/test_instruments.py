import numpy as np
import astropy.units as u
from astropy.time import Time
from mskpy import KeplerState

from .. import integrator, particle, rundynamics, generators
from .. import instruments
from . import sim_radius_uniform  # noqa: F401


class TestPhotometer:
    def test_init(self):
        inst = instruments.Photometer(5 * u.arcsec)
        assert inst.data.shape == ()

    def test_integrate(self, sim_radius_uniform):  # noqa: F811
        inst = instruments.Photometer(3 * u.rad)
        inst.integrate(sim_radius_uniform)
        assert inst.data == 2000


class TestCamera:
    def test_centering(self):
        """This is a regression test for camera shape and centering (v1.2.0)."""

        date = Time("2024-11-01")
        comet = KeplerState([2 * u.au.to("km"), 0, 0], [0, 30, 30], date)

        pgen = particle.Coma(
            comet,
            date,
        )
        pgen.composition = particle.Geometric()
        pgen.radius = generators.Uniform(1, 10)
        pgen.age = generators.Delta(0)
        pgen.nparticles = 1

        sim = rundynamics.run(pgen, integrator.BulirschStoer(), seed=24)
        sim.observer = KeplerState(
            [0, u.au.to("km"), 0], [30, 0, 0], sim.params["date"]
        )
        sim.observe()

        camera = instruments.Camera(size=(101, 3), crpix=(51, 2))
        camera.integrate(sim)
        assert camera.data[1, 50] == 1
        assert camera.data.sum() == 1

        camera = instruments.Camera(size=(101, 3), crpix=(2, 51))
        camera.integrate(sim)
        assert camera.data.sum() == 0
