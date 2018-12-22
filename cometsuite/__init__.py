"""Comet dust dynamics, and other similar problems.

Modules
-------
generators
graphics
instruments
integrator
particle
projection
rundynamics
scalers
simulation
state
templates
xyzfile

Examples
--------

Syndynes
^^^^^^^^

Generate a set of zero-ejection-velocity syndynes for comet Encke,
integrated with a two-body solution.  Save to an xyz file::

  >>> import cometsuite as cs
  >>> from mskpy import getspiceobj, KeplerState
  >>> 
  >>> jd = 2450643.5417
  >>> comet = KeplerState(getspiceobj('encke'), jd)
  >>> pgen = cs.Coma(comet, jd, composition=cs.Geometric())
  >>> 
  >>> beta = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.1]
  >>> ndays = 200
  >>> steps = 31
  >>> cs.syndynes(pgen, beta=beta, ndays=ndays, steps=steps)
  >>> 
  >>> integrator = cs.Kepler()
  >>> cs.run(pgen, integrator, xyzfile='iso.xyz')

If `xyzfile` is ommitted, `run` will return the results as a
`Simulation` object::

  >>> sim = cs.run(pgen, integrator)
  >>> print sim
  box: -1
  comet:
    kernel: None
    name: encke
    r:
    - 43691206.64150405
    - -164068210.11923924
    - -27400429.48185876
    v:
    - 26.404374475300806
    - -21.023864708842602
    - -1.633395548397659
  cometsuite: 1.0.0
  data:
  - d(radius)
  - d(graindensity)
  - d(beta)
  - d(age)
  - d[2](origin)
  - d[3](r_i)
  - d[3](v_ej)
  - d[3](r_f)
  date: '1997-07-14 01:00:02.880'
  header: 1.0
  integrator: Kepler(GM=1.3274935144e+20)
  label: None
  nparticles: 217
  pfunc:
    age: None
    composition: Geometric(rho0=1.0)
    density_scale: UnityScaler()
    radius: None
    speed: Delta(x0=0)
    speed_scale: UnityScaler()
    vhat: Sunward(pole=[ 0.  0.], w=None, distribution='uniformangle', theta_dist=Delta(x0=0),
      phi_dist=Delta(x0=0))
  save:
  - radius
  - graindensity
  - beta
  - age
  - origin
  - r_i
  - v_ej
  - r_f
  syndynes: true
  units:
  - micron
  - g/cm^3
  - none
  - s
  - deg
  - km
  - km/s
  - km
  data:
  - d(radius)
  - d(graindensity)
  - d(beta)
  - d(age)
  - d[2](origin)
  - d[3](r_i)
  - d[3](v_ej)
  - d[3](r_f)
  units:
  - micron
  - g/cm^3
  - none
  - s
  - deg
  - km
  - km/s
  - km
  ...
  
  Actual number of particles: 217
  Average log(beta): -2.20223839652
  Average beta: 0.0187142857143
  Average radius (micron): 175.207142857
  Average density (g/cm3): 1.0
  Average age (days): 100.0
  Average rh (AU): 0.969828705692

Then, plot the simulation::

  >>> from numpy import pi
  >>> import matplotlib.pyplot as plt
  >>> 
  >>> plt.clf()
  >>> ax = plt.subplot(polar=True, theta_offset=pi/2)
  >>> cs.synplot('iso.xyz')
  >>> plt.setp(ax, xlabel='Position angle', ylabel=r'$\\rho$ (arcsec)')
  >>> plt.draw()

For a more detailed example, see `cs.templates.example_syndynes`.

3D Coma
^^^^^^^

Simulate a coma for comet C/2009 P1 (Garradd)::

  >>> from mskpy import getspiceobj, KeplerState
  >>> import cometsuite as cs
  >>> import cometsuite.generators as g
  >>> import cometsuite.scalers as s
  >>> 
  >>> date = '2011-09-11'  # T-ReCS epoch
  >>> comet = KeplerState(getspiceobj('1003031'), date)
  >>> 
  >>> pgen = cs.Coma(comet, date)
  >>> pgen.composition = cs.Geometric(rho0=1)
  >>> pgen.age = g.Uniform(0, 365)
  >>> pgen.radius = g.Log(0, 3)
  >>> pgen.vhat = g.Isotropic()
  >>> pgen.speed = g.Delta(0.3)
  >>> pgen.speed_scale = s.SpeedRh() * s.SpeedRadius()
  >>> pgen.nparticles = 2000000
  >>> 
  >>> integrator = cs.Kepler()
  >>> cs.run(pgen, integrator, xyzfile='run01.xyz')

Then, visualize it::

  TBD

Simulate a comet break up
^^^^^^^^^^^^^^^^^^^^^^^^^

When comets split, the large fragments drift away with low velocities,
~5 m/s, in the radial and trasverse directions.  Run a simulation of
large (beta~0) particles with the particle functions::

  VELOCITY iso range 0.0 0.005 6
  LOGRADIUS 7 7
  Q_D iso
  LATITUDE 85 95
  POLE <lam> <bet>

The above parameters will eject 10-m sized fragments with velocities
from 0 to 5 m/s, near the orbital plane (LATITUDE is specified in
degrees from the pole).  To define the pole, get a vector
perpendicular to the orbital plane, and convert it into ecliptic
celestial coordinates::

  from mskpy1.spice import getxyz
  from mskpy1.observing import xyz2lb
  r, v = getxyz(<comet>, <date>)
  rcv = np.cross(r, v) / np.sqrt(np.sum(r**2)) / np.sqrt(np.sum(v**2))
  lam, bet = xyz2lb(rcv)  # degrees

where <date> is the epoch of fragmentation.  Read in the result, and
verify that the particles have low inclinations with respect to the
comet::

  from mskpy import Earth
  import cometsuite as cs
  sim = cs.Simulation('run.xyz', observer=Earth)
  vdrcv = np.sum(sim.v_ej * rcv) / sim.s_ej
  plangle = 90 - np.arccos(vdrcv)
  plangle[np.isnan(plangle)] = 0  # for v_ej = 0 particles
  print np.sum(np.abs(plangle) < np.radians(5)) / plangle.size

Now, plot their angular distances versus ejection velocity::

  import matplotlib.pyplot as plt
  plt.clf()
  ax = plt.gca()
  ax.scatter(sim.s_ej * 1000, sim.theta / 60., alpha=0.2)
  plt.setp(ax, xlabel='Ejection speed (m/s)', 'Angular distance (arcmin)')
  plt.draw()

The zero-ejection velocity particles are there to verify the accuracy
of the integration.  For example, splitting Comet C/2012 S1 (ISON) at
perihelion places zero-ejection velocity fragments 7.5 degrees away
from the ephemeris position of the comet.  This error is because at
this time (v0.9.5-dev) rundynamics does not include relativistic or
other effects relevant for a sungrazer.  Instead of comparing the
fragments to the comet's true ephemeris position, it is more relevant
to compare them to a zero-ejection velocity particle::

  lb0 = xyz2lb(sim.r_f[(sim.s_ej == 0)[0]])
  p = cs.Projection(lb0, [sim.lam, sim.bet], sim.Delta)
  plt.clf()
  ax = plt.gca()
  ax.scatter(sim.s_ej * 1000, p.theta / 60., alpha=0.2)
  plt.setp(ax, xlabel='Ejection speed (m/s)', 'Angular distance (arcmin)')
  plt.draw()

"""

__version__ = '1.0.0'

from . import simulation
from . import xyzfile
from . import projection
from . import instruments
from . import graphics

from . import state
from . import particle
from . import generators
from . import scalers
from . import integrator
from . import templates

from .simulation import *
from .xyzfile import *
from .projection import Projection
from .instruments import *
from .graphics import *

from .particle import *
from .rundynamics import run
from .integrator import *
from .templates import *
