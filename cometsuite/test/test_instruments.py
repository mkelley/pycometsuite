import astropy.units as u
from .. import instruments
from . import sim_radius_uniform


class TestPhotometer:
    def test_init(self):
        inst = instruments.Photometer(5 * u.arcsec)
        assert inst.data.shape == ()

    def test_integrate(self, sim_radius_uniform):
        inst = instruments.Photometer(5 * u.arcsec)
        inst.integrate(sim)
        assert inst.data == 2000
