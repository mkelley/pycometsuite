import numpy as np
from astropy.time import Time
from mskpy.ephem import KeplerState, getspiceobj, Earth
from ..templates import quick_syndynes


class TestQuickSyndynes:
    def test_object_state_vs_string(self):
        date = Time(2450643.5417, format="jd")
        opts = {
            "beta": [0.1],
            "ndays": 1,
            "steps": 1,
            "observer": KeplerState(Earth, date),
        }

        encke = KeplerState(getspiceobj("encke"), date)
        syn1 = quick_syndynes("encke", date, **opts)
        syn2 = quick_syndynes(encke, date, **opts)

        assert np.allclose(syn1.r_f, syn2.r_f)
