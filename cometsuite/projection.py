import numpy as np

class Projection(object):
    """Describes a set of positions projected onto the Celestial Sphere.

    Parameters
    ----------
    target : 2-element array
      The Celestial coordinates of the target: ra/dec, lam/bet,
      etc. [radians]
    particles : 2xN array
      The Celestial coordinates of the particles: ra/dec, lam/bet,
      etc. [radians]
    Delta : N-element array
      The observer-particle distances. [km]
    observer : SolarSysObject, optional
      The observer making the observation (for bookkeeping).

    Attributes
    ----------

    target
    particles
    Delta
    observer

    offset
    theta
    phi
    rho

    """

    def __init__(self, target, particles, Delta, observer=None):
        self.target = np.array(target).reshape((2, 1))
        self.particles = np.array(particles)
        self.Delta = Delta
        self.observer = observer

    def __repr__(self):
        return """Projection(target={},
            observer={})""".format(self.target.ravel(), self.observer)

    def __getitem__(self, i):
        if isinstance(i, str):
            return getattr(self, i)
        return Projection(self.target, self.particles[:, i], self.Delta[i],
                          observer=self.observer)

    @property
    def offset(self):
        """Projected (gnomonic) celestial coordinate offsets from the target [degrees]."""
        from numpy import degrees, cos
        from mskpy.util import hav, archav
        d = self.particles - self.target
        s = np.sign(d[0])
        d[0] = (s * archav(cos(self.particles[1]) * cos(self.target[1]) *
                           hav(d[0])))
        return degrees(d)

    @property
    def theta(self):
        """Angular distance to the target [arcsec]."""
        from numpy import degrees, cos
        from mskpy.util import hav, archav
        d = self.particles - self.target
        return degrees(archav(cos(self.particles[1]) * cos(self.target[1]) *
                              hav(d[0]) + hav(d[1]))) * 3600.0

    @property
    def phi(self):
        """Position angle w.r.t. the target, measured w.r.t. north [degrees]."""
        from numpy import degrees, sin, cos, tan, arctan2
        d = self.particles - self.target
        y = sin(d[0])
        x = (cos(self.target[1]) * tan(self.particles[1]) -
             sin(self.target[1]) * cos(d[0]))
        return degrees(arctan2(y, x))

    @property
    def rho(self):
        """Projected distance to target (gnomonic projection) [km]."""
        from numpy import cos, tan
        from mskpy.util import hav, archav
        d = self.particles - self.target
        rho = archav(cos(self.particles[1]) * cos(self.target[1]) *
                     hav(d[0]) + hav(d[1]))
        return 2 * self.Delta * tan(rho / 2)
