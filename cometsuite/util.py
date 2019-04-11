"""
General support functions.

Note: When new generators are added, those that may be used by
`ParticleGenerator`s should also be added to `ParticleGenerator.reset`.

Methods
-------
vector_rotate

"""

__all__ = [
    'vector_rotate'
]

import numpy as np


def lb2xyz(lon, lat):
    """Spherical coordinates to Cartesian."""
    clat = np.cos(lat)
    return np.array([clat * np.cos(lon),
                     clat * np.sin(lon),
                     np.sin(lat)])


def xyz2lb(r):
    """Cartesian to spherical coordinates (radians)."""
    lam = np.arctan2(r[1], r[0])
    bet = np.arctan2(r[2], np.sqrt(r[0]**2 + r[1]**2))
    return lam, bet


def magnitude(v):
    return np.sqrt(np.dot(v, v))


def spherical_rot(lam0, bet0, lam1, bet1, lam, bet):
    """Rotate coordinate by exmaple.

    Rotate lam, bet in the same way that lam0, bet0 needs to be
    rotated to match lam1, bet1.  All angles in radians.

    Based on the IDL routine spherical_coord_rotate.pro written by
    J.D. Smith, and distributed with CUBISM.

    """

    if (lam0 == lam1) and (bet0 == bet1):
        return (lam, bet)

    v0 = lb2xyz(lam0, bet0)
    v1 = lb2xyz(lam1, bet1)
    v = lb2xyz(lam, bet)

    # construct coordinate frame with x -> ref point and z -> rotation
    # axis
    x = v0
    z = np.cross(v1, v0)  # rotate about this axis
    z = z / magnitude(z)  # normalize
    y = np.cross(z, x)
    y = y / magnitude(y)

    # construct a new coordinate frame (x along new direction)
    x2 = v1
    y2 = np.cross(z, x2)
    y2 = y2 / magnitude(y2)

    # project onto the inital frame, the re-express in the rotated one
    if len(v.shape) == 1:
        v = (v * x).sum() * x2 + (v * y).sum() * y2 + (v * z).sum() * z
    else:
        vx = np.dot(v.T, x)
        vy = np.dot(v.T, y)
        vz = np.dot(v.T, z)
        v = vx * np.repeat(x2, v.shape[1]).reshape(v.shape)
        v += vy * np.repeat(y2, v.shape[1]).reshape(v.shape)
        v += vz * np.repeat(z,  v.shape[1]).reshape(v.shape)

    bet_new = np.arcsin(v[2])
    lam_new = np.arctan2(v[1], v[0]) % (2 * np.pi)

    return (lam_new, bet_new)


def vector_rotate(v, v_from, v_to):
    """Rotate `v` the same way that `v_from` should be rotated to match `v_to`.

    Parameters
    ----------
    v : array-like
      The vector(s) to rotate, shape = (3,) or (N, 3).
    v_from : array-like
      The initial reference vector(s), (3,) or (N, 3).
    v_to : array-like
      The final reference vectors(s), (3,) or (N, 3).

    Result
    ------
    r : ndarray
      The rotated vector(s), (3,) or (N, 3).

    """

    from mskpy.util import mhat

    v = np.array(v)
    v_from = np.array(v_from)
    v_to = np.array(v_to)

    if np.allclose(v_from, v_to):
        return v

    # algorithms assume Nx3 arrays
    if v.ndim == 1 and v_from.ndim == 1 and v_to.ndim == 1:
        return_single = True
    else:
        return_single = False

    if v.ndim == 1:
        v = v[np.newaxis]
    if v_from.ndim == 1:
        v_from = v_from[np.newaxis]
    if v_to.ndim == 1:
        v_to = v_to[np.newaxis]

    # we have Nx3 arrays, make sure all N are useable
    N = max([x.shape[0] for x in (v, v_from, v_to)])
    if (v.shape[0] not in (1, N) or v_from.shape[0] not in (1, N)
            or v_to.shape[0] not in (1, N)):
        raise ValueError('inputs must be (3,) or (N, 3), where N '
                         'matches all other inputs.')

    # Frame 1, x is the reference point, z is the rotation axis
    if v_from.shape[0] != N:
        x1 = np.tile(v_from, N).reshape((N, 3))
    else:
        x1 = v_from

    z1 = mhat(np.cross(v_from, v_to))[1]
    if z1.shape[0] != N:
        z1 = np.tile(z1, N).reshape((N, 3))

    y1 = mhat(np.cross(z1, x1))[1]
    f1 = np.dstack((x1, y1, z1))  # Nx3x3

    # Frame 2: x is the new reference point (axis), z is the same as
    # in Frame 1
    if v_to.shape[0] != N:
        x2 = np.tile(v_to, N).reshape((N, 3))
    else:
        x2 = v_to

    y2 = mhat(np.cross(z1, x2))[1]
    f2 = np.dstack((x2, y2, z1))  # Nx3x3

    # project onto Frame 1, re-express in Frame 2
    r = ((f1 * v[:, :, np.newaxis]).sum(1)[:, np.newaxis] * f2).sum(2)  # Nx3
    return r[0] if return_single else r


def gaussian(x, mu, sigma):
    """A normalized Gaussian function.


    Parameters
    ----------
    x : array
        Dependent variable.

    mu : float
        Position of the peak.

    sigma : float
        Width of the curve (sqrt(variance)).


    Returns
    -------
    G : ndarray

    """
    return (np.exp(-(x - mu)**2 / 2.0 / sigma**2) /
            np.sqrt(2.0 * np.pi) / sigma)
