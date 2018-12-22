"""
Keplerian motion.
"""

from .core import Integrator


class Kepler(Integrator):
    """Keplerian (two-body) motion.

    The solution is computed via a universal variables method in by
    the NAIF SPICE package.

    The default behavior is for orbits around the Sun.

    Paramters
    ---------
    M : float or Quantity, optional
      Mass of the central object.  [float: kg]
    GM : float or Quantity, optional
      Gravitational constant times the mass of the central
      object. [float: km**3/s**2]

    """

    def __init__(self, M=None, GM=None):
        import astropy.constants
        if M is None and GM is None:
            self.M = astropy.constants.M_sun
        elif M is not None:
            self.M = M
        else:
            self.GM = GM

    def __str__(self):
        return "Kepler(GM={})".format(self.GM)

    @property
    def M(self):
        return self._GM / 6.67384e-20

    @M.setter
    def M(self, m):
        import astropy.units as u
        if not isinstance(m, u.Quantity):
            m *= u.kg
        if not m.unit.is_equivalent(u.kg):
            raise u.UnitsError("M must have units of mass.")
        self._GM = 6.67384e-20 * m.to(u.kilogram).value

    @property
    def GM(self):
        return self._GM

    @GM.setter
    def GM(self, gm):
        import astropy.units as u
        if not isinstance(gm, u.Quantity):
            gm *= u.km**3 / u.s**2
        if not gm.unit.is_equivalent(u.km**3 / u.s**2):
            raise u.UnitsError("GM must have units of length**3 / time**2.")
        self._GM = gm.to(u.km**3 / u.s**2).value

    # def integrate(self, init, dt, beta=0):
    #    import numpy as np
    #    from spiceypy import prop2b
    #    from ..state import State

    #    rv = np.array(prop2b(self.GM * (1 - beta), init.rv, dt))
    #    return State(rv[:3], rv[3:], init.t + dt / 86400.0)

    def integrate(self, init, dt, beta=0):
        import numpy as np
        from ..state import State
        from .prop2b import prop2b

        rv = np.array(prop2b(self.GM * (1 - beta), init.rv, dt))
        return State(rv[:3], rv[3:], init.t + dt / 86400.0)

    integrate.__doc__ = Integrator.integrate.__doc__
