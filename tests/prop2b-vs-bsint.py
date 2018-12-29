import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import cometsuite as cs
from mskpy import getspiceobj, KeplerState, Earth


class Large(cs.particle.Composition):
    def __init__(self):
        self.rho0 = 1
        pass

    def beta(self, radius, porosity):
        return 0

    def radius(self, radius, porosity):
        return 1e12

    def __str__(self):
        return "Large()".format()


GM = 132712440041.939400 * u.km**3 / u.s**2
comet = KeplerState(getspiceobj('243P'), '2018-12-15')
comet.GM = GM.value

pgen = cs.Coma(comet, comet.jd, composition=Large(),
               age=cs.generators.Sequence(np.arange(100)),
               nparticles=100)
simk = cs.run(pgen, cs.Kepler(GM=GM))

pgen = cs.Coma(comet, comet.jd, composition=Large(),
               age=cs.generators.Sequence(np.arange(100)),
               nparticles=100)
simbs = cs.run(pgen, cs.BulirschStoer())

d = simk.r_f - simbs.r_f
print(np.sqrt(np.sum(d**2, 1)))

d0 = simbs.r_f[0] - simbs.r_f
print(np.sqrt(np.sum(d0**2, 1)))
