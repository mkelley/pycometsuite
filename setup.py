#!/usr/bin/env python
from distutils.core import Extension
from setuptools import setup

ext = [
    Extension(
        name="cometsuite.integrator.bsint",
        sources=["cometsuite/integrator/bsint.pyx"],
        libraries=["gsl", "gslcblas"],
    ),
    Extension(
        name="cometsuite.integrator.prop2b",
        sources=["cometsuite/integrator/prop2b.pyx"],
    ),
]

setup(ext_modules=ext)
