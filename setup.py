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

# from glob import glob
# from setuptools import setup, Extension, find_packages
# from Cython.Build import cythonize

# if __name__ == "__main__":
#     setup(name='cometsuite',
#           version='1.0.0',
#           description='Comet dust dynamics.',
#           author="Michael S. P. Kelley",
#           author_email="msk@astro.umd.edu",
#           packages=find_packages(),
#           scripts=glob('scripts/*'),
#           data_files=[('cometsuite/data', glob('cometsuite/data/*dat'))],
#           ext_modules=cythonize([
#               'cometsuite/integrator/prop2b.pyx',
#               Extension('cometsuite.integrator.bsint',
#                         ['cometsuite/integrator/bsint.pyx'],
#                         libraries=['gsl', 'gslcblas'])
#           ]),
#           license='BSD',
#           )
