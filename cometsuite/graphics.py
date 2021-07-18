import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as mpl

from mskpy.ephem import Earth
from mskpy.util import nearest, timesten

from .simulation import Simulation
from .xyzfile import XYZFile


def synplot(sim, rlim=None, synchrone=False,
            camera=None, offset=[0, 0], observer=None,
            betas=None, ages=None, labels=None, tmark=None,
            tanno=False, interp=False,
            ax=None, **kwargs):
    """Plot RunDynamics syndynes.

    The default is to plot syndynes/chrones in Celestial Coordinates
    for a polar plot. [arcseconds and degrees].


    Parameters
    ----------
    sim : string or Simulation
        The syndyne simulation.

    rlim : float, optional
        Don't plot points outside of ``rlim`` from the comet.  Set to ``None``
        for no limit.  [units: same as plot]

    synchrone : bool, optional
        Set to ``True`` to plot synchrones instead of syndynes.

    observer : SolarSysObject
        The observer, who observes the comet.  Default: Earth

    camera : Camera
        Rather than plotting absolute sky coordinates, plot pixel
        coordinates, as observed by ``camera``.

    offset : array_like (float, float), optional
        Offset the syndynes/chrones with these (dx, dy) values. [units: same
        as plot axes]

    betas : array_like, optional
        Limit the syndynes to these beta values.

    ages : array_like, optional
        Limit the synchrones to these ages.

    labels : array-like, strings, optional
        Use these labels for the lines.

    tmark : array-like, optional
        An array of ages at which to mark a vertical line [units: days].
        The program will mark the closest data point possible.

    tanno : bool or array-like, optional
        Set to True to annotate ``tmarks`` with thier ages; or set to an
        array of labels.

    interp : bool, optional
        Set to True to spline interpolate the data.  ``tmarks`` will
        continue to use the original data.

    ax : optional
        The axes to which to plot.

    **kwargs : optional
        Any matplotlib.plot() keyword argument.


    Returns
    -------
    lines : list
      A list of Matplotlib lines (or markers).


    Examples
    --------

    >>> from numpy import pi
    >>> import matplotlib.pyplot as plt
    >>> import cometsuite as cs
    >>> plt.clf()
    >>> ax = plt.subplot(polar=True, theta_offset=pi/2)
    >>> cs.synplot('syn.xyz')
    >>> ax.set_rmax(90)
    >>> labels = plt.setp(ax, xlabel='Position angle', ylabel=r'$\\rho$ (arcsec)')
    >>> labels[1].set_rotation(0)
    >>> plt.tight_layout()


    >>> from numpy import pi
    >>> import matplotlib.pyplot as plt
    >>> from astropy.io import fits
    >>> import cometsuite as cs
    >>> plt.clf()
    >>> ax = plt.subplot()
    >>> im, h = fits.getdata('image.fits', header=True)
    >>> cs.synplot('syn.xyz', camera=cs.Camera(fitsheader=h))
    >>> plt.setp(ax, xlabel='ΔRA (arcsec)', ylabel='ΔDec (arcsec)')
    >>> plt.tight_layout()

    """
    if observer is None:
        observer = Earth

    if isinstance(sim, str):
        xyzfile = sim
        sim = Simulation(xyzfile, observer=observer)

    if camera is None:
        xy = np.vstack((np.radians(sim.sky_coords.phi), sim.sky_coords.theta))
    else:
        camera.sky2xy(sim)
        xy = np.vstack((sim.x, sim.y))

    if ax is None:
        ax = mpl.gca()

    if synchrone:
        syns = _get_synchrones(sim, xy, ages, labels, rlim)
    else:
        syns = _get_syndynes(sim,  xy, betas, labels, rlim)

    lines = []
    for x, y, t, label in syns:
        # interpolate
        if interp:
            n = x.size
            x = splev(np.arange(n * 10) / 10.0, splrep(np.arange(n), x))
            y = splev(np.arange(n * 10) / 10.0, splrep(np.arange(n), y))

        lines.append(ax.plot(x, y, label=label, **kwargs)[0])

        if tmark is not None:
            c = lines[-1].get_color()
            for i in range(len(tmark)):
                j = nearest(tmark[i] * 86400, t)
                print(tmark[i], t[j])
                ax.plot([x[j]], [y[j]], marker='|', color=c)
                if tanno is not False:
                    if np.iterable(tanno):
                        s = tanno[i]
                    else:
                        s = t[j]
                    ax.annotate(s, (x[j], y[j]), color=c)

    return lines


def _get_syndynes(sim, xy, betas, labels, rlim):
    if betas is None:
        betas = np.unique(sim.beta)
        betas.sort()
    else:
        betas = np.array(betas)

    if labels is None:
        labels = [timesten(x, 3) for x in betas]

    for i in range(len(betas)):
        # take each beta to plot, sort by age
        j = np.flatnonzero(sim.beta == betas[i])
        j = j[sim.age[j].argsort()[::-1]]

        # limit to rlim?
        if rlim is not None:
            j = j[sim.sky_coords.theta[i] * 3600 <= rlim]

        x = xy[0, j]
        y = xy[1, j]
        t = sim.age[j]
        yield x, y, t, labels[i]


def _get_synchrones(sim, xy, ages, labels, rlim):
    if ages is None:
        ages = np.unique(sim.beta)
        ages.sort()
    else:
        ages = np.array(ages)

    if labels is None:
        labels = [timesten(x, 3) for x in ages]

    for i in range(len(ages)):
        # take each beta to plot, sort by age
        j = np.flatnonzero(np.isclose(sim.age, ages[i]))
        j = j[sim.beta[j].argsort()[::-1]]

        # limit to rlim?
        if rlim is not None:
            j = j[sim.sky_coords.theta[i] * 3600 <= rlim]

        x = xy[0, j]
        y = xy[1, j]
        b = sim.beta[j]
        yield x, y, b, labels[i]


def xyzplot3d(xyzfile):
    from enthought.mayavi import mlab
    r = xyzread(xyzfile, datalist=('r_f',))['r_f']
    r = r.T
    mlab.points3d(r[0], r[1], r[2], mode='point', colormap='hot')
    mlab.show()


def xyzplotvolume(xyzfile):
    from enthought.mayavi import mlab
    r = xyzread(xyzfile, datalist=('r_f',))['r_f']
    r -= r.mean(0)
    h = histogramdd(r.T, bins=30)
    v = h[0] / h[0].max()
    mlab.pipeline.volume(mlab.pipeline.scalar_field(v), vmax=0.8)
    mlab.show()
