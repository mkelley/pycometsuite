"""
Instruments for observing simulations.

"""

import numpy as np

__all__ = [
    'Instrument',
    'Camera',
    'Photometer',
]


class InvalidAxis(Exception):
    pass


class Instrument(object):
    """Simulated instruments that can record anything.

    Essentially a fancy histogram designed for `Simulation`.

    A set of canned instruments are defined in `cometsuite.instruments`.

    Parameters
    ----------
    axes : tuple
      Each axis is the name of a Simulation property that returns a
      single vector (e.g., radius, beta, s_ej, lam, bet, theta), a
      function that will generate data from the simulation, or a
      string describing a function to be passed to eval.  In the
      latter case, numpy is imported as np, the simulation is named
      sim, and the string must be prefixed with "func:".  For example,
      to return the heliocentric ecliptic x coordinate, use
      "func:sim.r_f[0]", or return the log of the radius with
      "func:np.log10(sim.radius)".
    scaler : Scaler or CompositeScaler, optional
      Particle scaler.
    normalizer : Scaler or CompositeScaler, optional
      Use this scaling function for computing the normalization array
      `n`.
    bins : int or ndarray, or tuples thereof, optional
      The number of bins or the bin edges for each axis.
    range : 2-tuple, or tuples thereof, optional
      When bin edges are not defined, use `range` as the left-most
      and right-most edges.

    Attributes
    ----------
    n : ndarray, int
      The number of particles that have been observed.
    data : ndarray, float
      The integration over all observed particles.
    extent : ndarray
      The left- and right-most edges of the arrays, e.g., for use with
      matplotlib's `imshow`.
    normalized : ndarray, float
      data / n.

    """

    def __init__(self, axes, scaler=None, normalizer=None, bins=10, range=None):
        from . import scalers
        self.axes = axes
        self.ndim = len(axes)
        self.shape = None
        self.n = None
        self.data = None
        self.reset(bins=bins, range=range)
        self.scaler = scalers.UnityScaler() if scaler is None else scaler
        self.normalizer = (
            scalers.UnityScaler() if normalizer is None else normalizer)

    def __repr__(self):
        size = 'x'.join([str(x) for x in self.shape])
        return "Instrument:\n  axes: {}\n  size: {}".format(
            ', '.join(self.axes), size)

    @property
    def extent(self):
        return [x[[0, -1]] for x in self.bins]

    @property
    def normalized(self):
        norm = self.data / self.n
        norm[self.n == 0] = 0
        return norm

    def reset(self, bins=None, range=None):
        """Initialize data arrays, optionally recomputing the bins.

        Parameters
        ----------
        bins : int or array, optional
          The number of bins or bin edges.
        range : array, optional
          When bins is an integer, use `range` as the left and right
          edges.

        """
        if (bins is None) and (range is None):
            # do nothing
            pass
        else:
            self.bins = np.histogramdd(np.zeros((1, self.ndim)),
                                       bins=bins, range=range)[1]
        self.shape = tuple([len(x) - 1 for x in self.bins])
        self.n = np.zeros(self.shape)
        self.data = np.zeros(self.shape)

    def integrate(self, sim):
        """Collect data from a simulation."""
        v = []
        for ax in self.axes:
            if hasattr(ax, '__call__'):
                v += [ax(sim)]
            elif isinstance(ax, str):
                if hasattr(sim, ax):
                    v += [sim[ax]]
                elif ax.startswith('func:'):
                    v += [eval(ax[5:])]
                else:
                    raise InvalidAxis(ax)
            else:
                raise InvalidAxis(ax)

            if len(v[-1]) != len(sim):
                raise InvalidAxis(
                    "Length of axis is {}, but was expected to be {}."
                    .format(len(v[-1]), len(sim)))

        w = self.scaler.scale(sim)
        n = self.normalizer.scale(sim)
        self.n += np.histogramdd(v, bins=self.bins, weights=n)[0]
        self.data += np.histogramdd(v, bins=self.bins, weights=w)[0]


class Camera(Instrument):
    """A CometSuite instrument that only takes images of the sky.

    Integrate converts sky coordinates to pixels, and ensures that the
    WCS is centered on the target, when needed.

    Usage::
      camera = Camera()
      camera = Camera(shape=shape, scale=scale, [center=], [centeryx=])
      camera = Camera(wcs=wcs, [shape=])
      camera = Camera(fitsheader=h)

    Parameters
    ----------
    shape : array, optional
      The y, x size of the camera. [pixels]
    scale : array, optional
      The spatial dispersion of the camera. [arcsec/pixel]
    center : array, optional
      The center of the field of view.  If None, the FOV will be
      centered on the target when the simulation is loaded.
    centeryx : array, optional
      Center the WCS on this pixel.

    wcs : astropy.wcs.WCS, optional
      The WCS of the image.

    fitsheader : string or astropy.io.fits.Header, optional
      A FITS header from which `wcs` and image size may be defined.

    scaler : Scaler or CompositeScaler, optional
      A particle scaler.
    normalizer : Scaler or CompositeScaler, optional
      Use this scaling function for computing the normalization array
      `n`.
    axes : array, optional
      Additional axes to add to the camera.  The first two axes will
      always be 'y' and 'x'.  See `Instrument` for details.
    bins : array, optional
      Specify the bins for the additional axes.  See `Instrument` for
      details.
    range : array, optional
      Specify the range for the additional axes.  See `Instrument` for
      details.

    Attributes
    ----------
    axes
    bins
    range
    shape
    scale
    center
    centeryx
    wcs
    fitsheader
    scaler
    normalizer

    n : ndarray
      The normalization array, which is by default the number of
      particles per pixel.
    data : ndarray
      The observation data.

    """

    def __init__(self, shape=(1024, 1024), scale=(-1, 1), center=None,
                 centeryx=None, wcs=None, fitsheader=None, scaler=None,
                 normalizer=None, axes=None, bins=None, range=None):

        from astropy.wcs import WCS

        self.center = center
        if fitsheader is not None:
            self.wcs = WCS(fitsheader)
            self.shape = (fitsheader['NAXIS1'], fitsheader['NAXIS2'])
        elif wcs is not None:
            self.wcs = wcs
            self.shape = shape
        else:
            self.shape = shape
            self.scale = scale
            if centeryx is None:
                centeryx = np.array(self.shape) / 2.0

            self.wcs = WCS(naxis=2)
            self.wcs.wcs.crpix = centeryx
            self.wcs.wcs.cdelt = np.array(self.scale) / 3600.0
            self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            if self.center is not None:
                self.wcs.wcs.crval = self.center

        _axes = ('y', 'x')
        _bins = [np.arange(self.shape[0] + 1, dtype=int),
                 np.arange(self.shape[1] + 1, dtype=int)]
        if axes is not None:
            # add additional axes to the camera
            _axes += tuple(axes)
            bins = np.histogramdd(np.zeros((1, len(axes))),
                                  bins=bins, range=range)[1]
            _bins.extend(bins)

        Instrument.__init__(self, _axes, scaler=scaler,
                            normalizer=normalizer, bins=_bins)

    @property
    def centeryx(self):
        xy = self.wcs.wcs_world2pix(self.center[0], self.center[1], 0)
        return np.array(xy[::-1])

    def sky2xy(self, sim):
        """Compute a simluation's particle's xy coordinates.
        """
        from numpy import degrees
        if self.center is None:
            # use the simulation to define the center of the FOV
            self.center = degrees(sim.sky_coords.target).flatten()
            self.wcs.wcs.crval = self.center
        x, y = np.array(self.wcs.wcs_world2pix(sim.lam, sim.bet, 0))
        dtype = [('x', float), ('y', float)]
        sim.array_coords = np.recarray(x.shape, dtype=dtype)
        sim.array_coords.x = x
        sim.array_coords.y = y

    def integrate(self, sim, reobserve=True):
        if (sim.array_coords is None) or reobserve:
            self.sky2xy(sim)
        i = (sim.x >= 0) * (sim.x < self.shape[0])
        i *= (sim.y >= 0) * (sim.y < self.shape[1])
        Instrument.integrate(self, sim[i])

    def write(self, filename, normalized=False, **keywords):
        """Write a FITS file, complete with WCS.

        2 extensions are saved:
          [0] The data (possibly normalized)
          [1] The number of particles per pixel.

        Parameters
        ----------
        filename : string
          The file name.
        normalized : bool
          Set to True to save normalized data.
        **keywords
          fits.HDUList.writeto() keywords.

        """

        from astropy.io import fits

        data = fits.HDUList()
        h = self.wcs.to_header()

        save = self.normalized if normalized else self.data
        n = self.n
        if save.ndim > 2:
            for i in range(save.ndim - 2):
                save = np.rollaxis(save, -1)
                n = np.rollaxis(n, -1)

        data.append(fits.PrimaryHDU(save, h))
        data.append(fits.ImageHDU(n, h))
        data.writeto(filename, **keywords)


class Photometer(Instrument):
    """A single-entrance aperture photometer.

    Integrate converts sky coordinates to radial offsets.

    Usage::
      phot = Photometer(rap)

    Parameters
    ----------
    rap : Quantity, optional
      The angula radius of the entrance aperture.

    scaler : Scaler or CompositeScaler, optional
      A particle scaler.
    normalizer : Scaler or CompositeScaler, optional
      Use this scaling function for computing the normalization array
      `n`.
    axes : array, optional
      Additional axes to add to the camera.  See `Instrument` for
      details.
    bins : array, optional
      Specify the bins for the additional axes.  See `Instrument` for
      details.
    range : array, optional
      Specify the range for the additional axes.  See `Instrument` for
      details.

    Attributes
    ----------
    axes
    bins
    range
    center
    scaler
    normalizer

    n : ndarray
      The normalization array, which is by default the number of
      particles per pixel.
    data : ndarray
      The observation data.

    """

    def __init__(self, rap, scaler=None, normalizer=None,
                 axes=None, bins=None, range=None):

        import astropy.units as u

        assert rap.unit.is_equivalent(u.arcsec)
        self.rap = rap.to(u.arcsec).value

        _axes = []
        _bins = []
        if axes is not None:
            # add additional axes
            _axes += tuple(axes)
            bins = np.histogramdd(np.zeros((1, len(axes))),
                                  bins=bins, range=range)[1]
            _bins.extend(bins)

        Instrument.__init__(self, _axes, scaler=scaler,
                            normalizer=normalizer, bins=_bins)

    def integrate(self, sim):
        i = sim.theta < self.rap
        Instrument.integrate(self, sim[i])


def ccd():
    """Generic CCD."""
    from .scalers import ScatteredLight
    return Camera(scaler=ScatteredLight(0.6),
                  shape=(1024, 1024), scale=(-1, 1))


def ircam():
    """Generic 10 um camera."""
    from .scalers import ThermalEmission
    return Camera(scaler=ThermalEmission(10),
                  shape=(1024, 1024), scale=(-1, 1))


def acs():
    """HST Advanced Camera for Surveys."""
    from .scalers import ScatteredLight
    return Camera(scaler=ScatteredLight(0.6),
                  shape=(4096, 4096), scale=(-0.05, 0.05))


def wfc3_uvis():
    """HST Wide Field Camera 3, UVIS detector."""
    from .scalers import ScatteredLight
    return Camera(scaler=ScatteredLight(0.6),
                  shape=(4096, 4096), scale=(-0.04, 0.04))


def di_mri():
    """Deep Impact Flyby spacecraft Medium Resolution Imager."""
    from .scalers import ScatteredLight
    return Camera(scaler=ScatteredLight(0.6),
                  shape=(1024, 1024), scale=(-2.1, 2.1))
