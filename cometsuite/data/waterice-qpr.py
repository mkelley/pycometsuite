import numpy as np
from numpy import pi
from mskpy.calib import solar_flux
from mskpy.util import davint
import grains2
from grains2.bhmie import bhmie

ac = grains2.waterice()
vac = grains2.vacuum()
wave = np.linspace(0.25, 3, 100)
S_lam = solar_flux(wave).value

#w = np.logspace(np.log10(0.25), 1)
#S = davint(w, solar_flux(w).value, wave[0], wave[-1])

p = np.r_[np.linspace(0, 0.9, 31), np.arange(0.91, 1, 0.01)]
a = np.logspace(-1, 2.5, 200)
qpr = np.zeros((len(p), len(a)))
ema = grains2.Bruggeman()
for i in range(len(p)):
    m = ema.mix((ac, vac), (1 - p[i], p[i]))
    for j in range(len(a)):
        x = 2 * pi * a[j] / wave
        nk = m.ri(wave)
        qext, qsca, gsca = np.zeros((3, len(x)))
        for k in range(len(x)):
            q = bhmie(x[k], nk[k], 1)
            qext[k] = q[2]
            qsca[k] = q[3]
            gsca[k] = q[5]
        qbext = davint(wave, qext * S_lam, wave[0], wave[-1]) / 1366.1
        qbsca = davint(wave, qsca * S_lam, wave[0], wave[-1]) / 1366.1
        gbsca = davint(wave, gsca * S_lam, wave[0], wave[-1]) / 1366.1
        qpr[i, j] = qbext - gbsca * qbsca

save = [np.r_[-1, p]]
save.extend([np.r_[a[i], qpr[:, i]] for i in range(len(a))])
np.savetxt('waterice-qpr.dat', tuple(save),
           header='porosity = first row, size = first column')
