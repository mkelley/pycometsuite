import os
import argparse
import numpy as np
import astropy.units as u
import cometsuite as cs
import cometsuite.generators as g
import cometsuite.scalers as s
from mskpy import FixedState, lb2xyz

# 10,000 km radius isotropic coma
#target = FixedState(np.array([2, 0, 0]) * u.au.to(u.km))
#date = '2000-01-01'
# pgen = cs.Coma(
#    target, date,
#    composition=cs.Geometric(rho0=1),
#    radius=g.Delta(1e4),
#    age=g.Uniform(0, 10000 / 86400),
#    vhat=g.Isotropic(),
#    speed=g.Delta(1),
#    nparticles=100000
#)
#
#integrator = cs.BulirschStoer()
#cs.run(pgen, integrator, xyzfile='iso.xyz')

target = np.array([2, 0, 0]) * u.au.to(u.km)

params = {
    'box': -1,
    'comet': {
        'kernel': 'None', 'name': None,
        'r': list(target),
        'v': [0.0, 0.0, 0.0]
    },
    'cometsuite': '1.0.0',
    'data': ['d(radius)',
             'd(age)',
             'd(v_ej)',
             'd[3](r_f)'],
    'date': '2000-01-01 00:00:00.000',
    'header': 1.0,
    'integrator': 'None',
    'label': 'None',
    'nparticles': 100000,
    'pfunc': {
        'age': 'Uniform(x0=0, x1=0.11574074074074074)',
        'composition': 'Geometric(rho0=1)',
        'density_scale': 'UnityScaler()',
        'radius': 'Delta(x0=10000.0)',
        'speed': 'Delta(x0=1)',
        'speed_scale': 'UnityScaler()',
        'vhat': 'Isotropic()'
    },
    'save': ['radius',
             'age',
             'v_ej',
             'r_f'],
    'syndynes': False,
    'units': ['micron', 's', 'km/s', 'km']
}

sim = cs.Simulation(**params)

pi = np.pi
for i in range(params['nparticles']):
    sim.particles[i].age = np.random.rand() * 1e4
    phi = np.random.rand() * 2 * pi
    th = np.arccos(1 - 2 * np.random.rand()) - pi / 2
    sim.particles[i].radius = 10000
    v_ej = np.array((
        np.cos(th) * np.cos(phi),
        np.cos(th) * np.sin(phi),
        np.sin(th)
    ))
    sim.particles[i].v_ej = v_ej
    sim.particles[i].r_f = sim.particles[i].age * v_ej + target

sim.observer = FixedState(np.array([1, 0, 0]) * u.au.to(u.km))
sim.observe()
with cs.XYZFile('iso.xyz', 'w', sim) as outf:
    outf.write_particles(sim.particles)

camera = cs.Camera(shape=(300, 300),
                   scale=np.r_[-0.09, 0.09],
                   center=np.degrees(sim.sky_coords.target.flatten()))
camera.integrate(sim)
full = camera.data

# SkyCoord(0, 90, unit='deg', frame=coords.GeocentricTrueEcliptic).icrs
# <SkyCoord (ICRS): (ra, dec) in deg
#    (270.01429703, 66.56176469)>
os.system('python3 ../scripts/cs-explore-source iso.xyz -n24 -s300'
          ' --pixel-scale=0.09 --pole=270,66.6 --observer=1,0,0'
          ' --observer-units=au')
