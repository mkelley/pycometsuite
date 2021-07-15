import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as mpl

from mskpy.ephem import Earth
from mskpy.util import nearest, timesten

from .simulation import Simulation
from .xyzfile import XYZFile


def synplot(sim, rlim=None, synchrone=False,
            camera=None, offset=[0, 0], observer=None,
            betalist=None, agelist=None, labels=None, tmark=None,
            tanno=False, interp=False, color=True,
            plot3d=False, ax=None, silent=False, **kwargs):
    """Plot RunDynamics syndynes.

    The default is to plot syndynes/chrones in Celestial Coordinates
    for a polar plot. [arcseconds and degrees].

    Parameters
    ----------
    sim : string or Simulation
      The syndyne simulation.
    rlim : float, optional
      Don't plot points outside of ``rlim`` from the comet.  Set to
      None for no limit.  [units: same as plot]
    synchrone : bool, optional
      Set to True to plot synchrones instead of syndynes.
    plot3d : bool, optional
      Set to true for 3D plots.  When enabled, the syndynes will be
      plot using cometocentric ecliptic coordinates.
    observer : SolarSysObject
      The observer, who observes the comet.  Default: Earth
    camera : Camera
      Rather than plotting absolute sky coordinates, plot pixel
      coordinates, as observed by ``camera``.
    offset : array_like (float, float), optional
      Offset the syndynes/chrones with these (dx, dy) values [units:
      same as plot axes]
    betalist : array_like, optional
      Limit the syndynes to these beta values.
    agelist : array_like, optional
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
    color : bool, optional
      Set to True or a list of Matplotlib colors to plot syndynes in
      color.
    ax : optional
      The axes to which to plot.
    silent : bool, optional
      Set to True to prevent synplot() from printing runtime outputs.
    **kwargs : optional
      Any matplotlib.plot() keyword argument.

    Returns
    -------
    lines : list
      A list of Matplotlib lines (or markers).

    Examples
    --------

    from numpy import pi
    import matplotlib.pyplot as plt

    plt.clf()
    ax = plt.subplot(polar=True, theta_offset=pi/2)
    cs.synplot('syn.xyz')
    ax.set_rmax(90)
    labels = plt.setp(ax, xlabel='Position angle', ylabel=r'$\\rho$ (arcsec)')
    labels[1].set_rotation(0)
    plt.tight_layout()
    plt.draw()

    """
    if observer is None:
        observer = Earth

    if isinstance(sim, str):
        xyzfile = sim
        sim = Simulation(xyzfile, observer=observer, camera=camera)

    if camera is None:
        xy = np.vstack((np.radians(sim.sky_coords.phi), sim.sky_coords.theta))
    else:
        xy = np.vstack((sim.x, sim.y))

    lines = []

    if synchrone:
        pass
    else:
        def synitems(sim=sim, betalist=betalist, labels=labels):
            # syndynes
            if betalist is None:
                betalist = np.unique(sim.beta)
                betalist.sort()
            else:
                betalist = np.array(betalist)

            if labels is None:
                labels = [timesten(x, 3) for x in betalist]

            for i in range(len(betalist)):
                # take each beta to plot, sort by age
                j = np.flatnonzero(sim.beta == betalist[i])
                j = j[sim.age[j].argsort()[::-1]]

                # limit to rlim?
                if rlim is not None:
                    j = j[sim.sky_coords.theta[i] * 3600 <= rlim]

                x = xy[0, j]
                y = xy[1, j]
                t = sim.age[j]
                yield x, y, t, labels[i]

    if ax is None:
        ax = mpl.gca()

    for x, y, t, label in synitems():
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
