"""
State vectors.
"""

import numpy as np
import astropy.units as u

class State(object):
    def __init__(self, r, v, t):
        """State vector.

        Parameters
        ----------
        r, v : array
          Position and velocity.
        t : float
          Julian date.

        Attributes
        ----------
        r, v : ndarray
          Position and velocity. [km, km/s]
        t : float
          Julian date.
        et : float
          SPICE ephemeris time.

        """

        from mskpy.util import date2time

        self.r = np.array(r)
        self.v = np.array(v)
        self.t = np.array(t)

    @property
    def et(self):
        from mskpy.ephem.core import date2et
        return jd2et(self.t)

    @property
    def rv(self):
        """Position and velocity as a 6-element array."""
        return np.concatenate((self.r, self.v))

    def __str__(self):
        return "[ r={}\n  v={}\n  t={} ]".format(str(self.r), str(self.v), str(self.t))
