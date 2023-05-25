# Cometsuite

Simulate cometary dust dynamics.

## Requirements

* Python 3
  * numpy
  * scipy
  * Cython
  * matplotlib
  * [mskpy](https://github.com/mkelley/mskpy)
  * spiceypy
* GNU Scientific Library and GSL CBLAS library


## File formats

XYZ files are a saved particle simulation.  The files consist of two parts: the header and the particle array.  The header describes the simulation parameters and the particle array format.

### XYZ v1

v1 headers are valid YAML with the following keywords:

```
box: -1
comet:
kernel: None
name: C/2009 P1
r:
- 199512618.6570191
- -186957807.98389882
- 149658240.7133305
v:
- -23.28632377134447
- 10.860849832337532
- 13.85300854007757
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
date: '2011-09-11 00:00:00.000'
header: 1.0
integrator: Kepler(GM=1.3274935144e+20)
label: None
nparticles: 10000000
pfunc:
age: Uniform(x0=0, x1=31536000)
composition: Geometric(rho0=1)
density_scale: UnityScaler()
radius: Log(x0=0, x1=3)
speed: Delta(x0=0.1)
speed_scale: SpeedRh(k=-0.5) * SpeedRadius(k=-0.5)
vhat: Isotropic()
save:
- radius
- graindensity
- beta
- age
- origin
- r_i
- v_ej
- r_f
syndynes: false
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
```

### XYZ v0

Older versions of the file format can be read with `cometsuite.xyzfiles.XYZFile0`.  The v0.x headers are key-value pairs, separated by a colon.  Comments start with a "#" character in the first column:

```
# CometSuite 0.9.5
# Valid program names: syndynes, make comet, integratexyz
PROGRAM: syndynes
# Parameters common to all programs.
COMET: encke
KERNEL: encke.bsp
JD: 2450643.5417
XYZFILE: output.xyz
LABEL: 
PFUNC: 
TOL: 0.01
PLANETS: 511
PLANETLOOKUP: 0
CLOSEAPPROACHES: 1
BOX: -1
LTT: 0
SAVE: radius graindensity beta age origin r_i v_ej r_f
# Syndyne specific section.
BETA: 0.001 0.002 0.004 0.006 0.008 0.01 0.1
NDAYS: 200
STEPS: 31
ORBIT: 1
# Make comet specific section.
NPARTICLES: 100000
# data file description
UNITS: micron g/cm^3 none s deg km/s km km/s s km km/s s
DATA: d(radius) d(graindensity) d(beta) d(age) d[2](origin) d[3](v_ej) d[3](r_i) d[3](v_i) d(t_i) d[3](r_f) d[3](v_f) d(t_f)
```


## Acknowledgements

Two-body orbit propagation code is based on the universal variables approach in the [SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html) from the Navigation and Ancillary Information Facility (NAIF) at the NASA Jet Propulsion Laboratory.

The RADAU15 integrator of Everhart (1985), rewritten in C for the original C++ version of Cometsuite is included in this distribution.  (Everhart 1985. in Dynamics of Comets: Their Origin and Evolution. A. Carusi and G. Valsecchi, eds. Astrophysics and Space Science Library 115 185).