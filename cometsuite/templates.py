"""
templates - CometSuite templates
================================

Functions
---------
example_coma
example_coma_parallel
exmaple_syndynes
quick_syndynes

"""

__all__ = [
    "example_coma",
    "example_coma_parallel",
    "example_syndynes",
    "quick_syndynes",
]

import os
import numpy as np
import matplotlib.pyplot as plt
from mskpy import getspiceobj, State, KeplerState, Earth, getgeom


def example_syndynes(filename):
    """Write an example syndyne script.

    Parameters
    ----------
    filename : string
      The name of the file to write.  Existing files will not be
      overwritten.

    """

    if os.path.exists(filename):
        raise IOError("File already exists: {}".format(filename))

    with open(filename, "w") as outf:
        synfilename = os.path.splitext(os.path.split(filename)[1])[0]
        outf.write(
            """import numpy as np
import cometsuite as cs
from mskpy import rarray, tarray
from mskpy import getspiceobj, KeplerState, Earth
import matplotlib.pyplot as plt

comet = KeplerState(getspiceobj('C/2013 A1'), '2014-01-21')
pgen = cs.Coma(comet, comet.jd, composition=cs.Geometric())

beta = np.logspace(-3, -1, 5)
ndays = 365
steps = 101
cs.syndynes(pgen, beta=beta, ndays=ndays, steps=steps)

integrator = cs.Kepler()
cs.run(pgen, integrator, xyzfile='{0}.xyz')
sim = cs.Simulation('{0}.xyz', observer=Earth)

# im = fits.getdata('comet.fits')
# yx0 = 383, 637
# platescale = 0.5
# r = rarray(im.shape, yx0) * platescale
# t = tarray(im.shape, yx0) - np.pi / 2

plt.clf()
ax = plt.subplot(polar=True, theta_offset=np.pi / 2)

cs.synplot(sim)
# ax.pcolormesh(t, r, im)

ax.set_rmax(60)
plt.setp(ax, xlabel='Position angle', ylabel=r'$\\rho$ (arcsec)')
ax.yaxis.label.set_rotation(0)
ax.legend(fontsize='medium', loc='center left',
          bbox_to_anchor=(1.1, 0.5))
plt.tight_layout(rect=(0, 0, 0.8, 1))
plt.draw()
""".format(
                synfilename
            )
        )


def quick_syndynes(
    obj,
    date,
    beta=None,
    ndays=365,
    steps=101,
    observer=None,
    integrator=None,
    align="north",
    **kwargs
):
    """Compute and plot syndynes.


    Parameters
    ----------
    obj : string or `~mskpy.ephem.state.State`
        The source object as a name, whose SPICE kernel can be retrieved via
        `mskpy.getspiceobj(obj)`, or as a ``State`` object.

    date : various
        The time of observation in a format acceptable to `mskpy.date2time`.

    beta : array, optional

    ndays : float, optional

    steps : int, optional
        Parameters for the generator setup function, `syndynes`.  If `beta` is
        `None`, a default set will be used.

    observer : SolarSysObject, optional
        The observer.  If `observer` is `None`, Earth will be used.

    integrator : Integrator, optional
        Use this integrator, default `Kepler`.

    align : string, optional
        Rotate the plot to align "north" to up, or "sun" to right.

    **kwargs
        Any `synplot` keywords.


    Returns
    -------
    sim : Simulation
        The syndyne simulation.

    """

    import cometsuite as cs

    if beta is None:
        beta = np.logspace(-3, 0, 7)
    if observer is None:
        observer = Earth

    if isinstance(obj, State):
        target = obj
    else:
        target = KeplerState(getspiceobj(obj), date)

    if integrator is None:
        integrator = cs.Kepler()

    pgen = cs.Coma(target, target.jd, composition=cs.Geometric())
    cs.syndynes(pgen, beta=beta, ndays=ndays, steps=steps)

    sim = cs.run(pgen, integrator)
    sim.observer = observer
    sim.observe()

    if align == "north":
        theta_offset = np.pi / 2
    elif align == "sun":
        geom = getgeom(getspiceobj(obj), observer, date)
        theta_offset = -geom.sangle.rad
    else:
        raise ValueError("Invalid `align`")

    plt.clf()
    ax = plt.subplot(111, polar=True, theta_offset=theta_offset)
    cs.synplot(sim, **kwargs)
    # ax.set_rmax()
    labels = plt.setp(ax, xlabel="Position angle", ylabel=r"$\rho$ (arcsec)")
    labels[1].set_rotation(0)
    ax.legend(
        prop=dict(size="medium"), loc="center left", bbox_to_anchor=(1.1, 0.5)
    )
    plt.tight_layout(rect=(0, 0, 0.6, 1))
    plt.draw()

    return sim


def example_coma(filename):
    """Write an example coma script.

    Parameters
    ----------
    filename : string
      The name of the file to write.  Existing files will not be
      overwritten.

    """
    from os.path import exists, split, splitext

    if exists(filename):
        raise IOError("File already exists: {}".format(filename))

    with open(filename, "w") as outf:
        xyzfilename = splitext(split(filename)[1])[0]
        outf.write(
            """from mskpy import getspiceobj, KeplerState
import cometsuite as cs
import cometsuite.generators as g
import cometsuite.scalers as s

date = '2011-09-11'  # T-ReCS epoch
comet = KeplerState(getspiceobj('1003031'), date)

pgen = cs.Coma(comet, date)
pgen.composition = cs.Geometric(rho0=1)
pgen.age = g.Uniform(0, 365)
pgen.radius = g.Log(0, 3)
pgen.vhat = g.Isotropic()
pgen.speed = g.Delta(0.3)
pgen.speed_scale = s.SpeedRh() * s.SpeedRadius()
pgen.nparticles = 2000000

integrator = cs.Kepler()
cs.run(pgen, integrator, xyzfile='{}.xyz')
""".format(
                xyzfilename
            )
        )


def example_coma_parallel(filename):
    """Write an example coma script.

    Parameters
    ----------
    filename : string
      The name of the file to write.  Existing files will not be
      overwritten.

    """
    from os.path import exists, split, splitext

    if exists(filename):
        raise IOError("File already exists: {}".format(filename))

    with open(filename, "w") as outf:
        xyzfilename = splitext(split(filename)[1])[0]
        outf.write(
            """from multiprocessing import Pool
from mskpy import getspiceobj, KeplerState
import cometsuite as cs
import cometsuite.generators as g
import cometsuite.scalers as s

def runsim(i):
        print('Queuing {{}}'.format(i))
    date = '2013-03-07'
    comet = KeplerState(getspiceobj('C/2012 S1'), date)

    pgen = cs.Coma(comet, date)
    pgen.composition = cs.Geometric(rho0=1)
    pgen.age = g.Uniform(0, 365)
    pgen.radius = g.Log(0, 3)
    pgen.vhat = g.Isotropic()
    pgen.speed = g.Delta(0.3)
    pgen.speed_scale = s.SpeedRh() * s.SpeedRadius()
    pgen.nparticles = 1000

    integrator = cs.Kepler()
    cs.run(pgen, integrator, xyzfile='{}{{:02d}}.xyz'.format(i))

with Pool() as pool:
    for i in range(4):
        pool.apply_async(runsim, (i,))

    pool.close()
    pool.join()
""".format(
                xyzfilename
            )
        )
