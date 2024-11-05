import astropy.units as u
from .. import instruments
from . import sim


class TestPhotometer:
    def test_init(self):
        inst = instruments.Photometer(5 * u.arcsec)
        assert inst.data.shape == ()

    def test_integrate(self, sim):
        inst = instruments.Photometer(5 * u.arcsec)
        inst.integrate(sim)
        assert inst.data == 2000
