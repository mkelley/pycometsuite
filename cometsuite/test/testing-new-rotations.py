import numpy as np
from mskpy import lb2xyz
import cometsuite as cs

N = 100
#theta_dist = cs.generators.UniformAngle()
theta_dist = cs.generators.Delta(np.pi / 2)
phi_dist = cs.generators.Uniform(0, 2 * np.pi)

# choose theta and phi, define radial vector, all w.r.t. axis
# of symmetry
theta = np.pi / 2 - theta_dist.next(N)
phi = phi_dist.next(N)
r = np.array((np.cos(theta) * np.cos(phi),
              np.cos(theta) * np.sin(phi),
              np.sin(theta)))

# define 100 axes of symmetry
axis = np.ones(300).reshape((3, 100))
axis = (axis / np.sqrt(np.sum(axis**2, 0)))

# define pole
pole = np.array((0, 0, 1.0))

# `r` is in the axis of symmetry reference frame
# we want it in planetographic coordinates

# rotate `r` into planetographic coordinates the same way we
# need to rotate pole into `axis`
# Frame1: x is the reference point (pole), z is the rotation axis
x1 = np.repeat(pole, axis.shape[1]).reshape(axis.shape)
z1 = np.cross(pole, axis.T).T
z1 /= np.sqrt(np.sum(z1**2, 0))
y1 = np.cross(z1, x1, axisa=0, axisb=0).T
y1 /= np.sqrt(np.sum(y1**2, 0))
f1 = np.dstack((x1.T, y1.T, z1.T))  # Nx3x3

# Frame2: x is the new reference point (axis)
x2 = axis
z2 = z1
y2 = np.cross(z2, x2, axisa=0, axisb=0).T
y2 /= np.sqrt(np.sum(y2**2, 0))
f2 = np.dstack((x2.T, y2.T, z2.T))

# Check work
xt1 = pole
zt1 = np.cross(pole, axis[:, 0]).T
zt1 /= np.sqrt(np.sum(zt1**2, 0))
yt1 = np.cross(zt1, xt1)
yt1 /= np.sqrt(np.sum(yt1**2, 0))
ft1 = np.dstack((xt1, yt1, zt1))  # 3x3

xt2 = axis[:, 0]
zt2 = zt1
yt2 = np.cross(zt2, xt2)
yt2 /= np.sqrt(np.sum(yt2**2))
ft2 = np.dstack((xt2, yt2, zt2))

vt = sum(ft1[0, :, 0] * r[:, 0]) * ft2[0, :, 0] + sum(ft1[0, :, 1] * r[:, 0]) * ft2[0, :, 1] + sum(ft1[0, :, 2] * r[:, 0]) * ft2[0, :, 2]

#v = np.dot(f2[0], np.dot(f1[0], r))
#vt = ft2 * np.dot(ft1, r[:, 0])

# project into Frame1, express as Frame2
#v = np.dot(x1, r) * x2 + np.dot(y1, r) * y2 + np.dot(z1, r) * z2
#
#v = (np.rollaxis(f1, 0, 3) * r).sum(1)
#v = (np.rollaxis(f2, 0, 3) * v).sum(1)

v = ((f1.T * r).sum(1) * np.rollaxis(f2, 0, 3)).sum(1)

assert np.allclose(f1[0], ft1)
assert np.allclose(f2[0], ft2)
assert np.allclose(v[:, 0], vt)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*r)
ax.plot(*zip([0, 0, 0], axis[:, 0]))
ax.plot(*zip([0, 0, 0], pole))
ax.scatter(*v)


########################################################################
# refactor for Nx3 vectors
r = np.c_[np.cos(theta) * np.cos(phi),
          np.cos(theta) * np.sin(phi),
          np.sin(theta)]

# define 100 axes of symmetry
axis = np.ones(300).reshape((100, 3))
axis /= np.sqrt(np.sum(axis**2, 1))[:, np.newaxis]

x1 = np.tile(pole, axis.shape[0]).reshape(axis.shape)
z1 = np.cross(pole, axis)
z1 /= np.sqrt(np.sum(z1**2, 1))[:, np.newaxis]
y1 = np.cross(z1, x1)
y1 /= np.sqrt(np.sum(y1**2, 1))[:, np.newaxis]
f1 = np.dstack((x1, y1, z1))  # Nx3x3

x2 = axis
z2 = z1
y2 = np.cross(z2, x2)
y2 /= np.sqrt(np.sum(y2**2, 1))[:, np.newaxis]
f2 = np.dstack((x2, y2, z2))

v = ((f1 * r[:, :, np.newaxis]).sum(1)[:, np.newaxis] * f2).sum(2)

assert np.allclose(f1[0], ft1)
assert np.allclose(f2[0], ft2)
assert np.allclose(v[0], vt)

# OK now we have a vector of stuff, project them onto the pole frame
pframe = cs.generators.Vej.pole2basis(pole)
p = (pframe * v[:, np.newaxis]).sum(2)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
fig.clear()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*r.T)
ax.plot(*zip([0, 0, 0], axis[0]))
ax.plot(*zip([0, 0, 0], pole))
ax.scatter(*v.T)
ax.scatter(*p.T)
