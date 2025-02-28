import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from mskpy import KeplerState, Earth, getspiceobj
from sbpy.calib import Sun
import cometsuite as cs
from cometsuite import particle, generators


date = Time("2018-12-12")
# comet Wirtanen outburst
comet = getspiceobj("46p")

pgen = particle.Coma(
    comet,
    date,
)
pgen.composition = particle.Geometric()
pgen.radius = generators.Log(-1, 3)
pgen.age = generators.Delta(0)
pgen.nparticles = 2000


xyzfile = "scattered-light.xyz"
if not os.path.exists(xyzfile):
    cs.run(pgen, cs.BulirschStoer(), xyzfile=xyzfile)

sim = cs.Simulation(xyzfile, observer=Earth)
rh = sim.rh_f.mean()
Delta = sim.Delta.mean()

wave = 0.63
sun = Sun.from_default()
# a little wavelength averaging to mitigate absorption line issues
w = np.r_[0.95, 1.0, 1.05] * wave * u.um
S = sun(w, unit="W/(m2 um)")[1]  # at 1 AU

psd = cs.scalers.PSD_PowerLaw(-3.5) * cs.scalers.PSD_RemoveLogBias()
light = cs.scalers.ScatteredLight(wave, Ap=0.05)

# calibrate to total mass of 1e6 kg
M_sim = (psd.scale(sim) * sim.m / 1e3).sum()  # kg
mass_cal = 1.6e6 / M_sim * psd.scale(sim)
sigma = mass_cal * np.pi * (1e-6 * sim.radius) ** 2  # m2

Fsim = mass_cal * light.scale(sim)
G = np.pi / 0.05 * Fsim.sum() * rh**2 / S.value * sim.Delta.mean() ** 2

# calibrate to total cross sectional area of 118 km2 = 118e6 m2
G_sim = (psd.scale(sim) * sim.cs).sum() / 1e4  # m2
area_cal = 118e6 / G_sim * psd.scale(sim)
M = (area_cal * sim.m).sum() / 1e3  # kg

# sigma = 118 * 1e6
F = 0.05 / np.pi * sigma * S / rh**2 / (1e3 * Delta) ** 2
k = wave / 2 / np.pi
x = sim.radius / k
i = x < 1
F[i] *= (sim.radius / k)[i] ** 4

m = F.sum().to(u.ABmag, u.spectral_density(wave * u.um))

i = np.argsort(sim.radius)

plt.clf()
# plt.plot(sim.radius[i], x[i])
plt.plot(sim.radius[i], F[i])
plt.plot(sim.radius[i], Fsim[i])
plt.xscale("log")
plt.yscale("log")
