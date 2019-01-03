#!/usr/bin/env python
from glob import glob
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

if __name__ == "__main__":
    setup(name='cometsuite',
          version='1.0.0',
          description='Comet dust dynamics.',
          author="Michael S. P. Kelley",
          author_email="msk@astro.umd.edu",
          packages=find_packages(),
          data_files=[('cometsuite/data', glob('cometsuite/data/*dat'))],
          ext_modules=cythonize([
              'cometsuite/integrator/prop2b.pyx',
              Extension('cometsuite.integrator.bsint',
                        ['cometsuite/integrator/bsint.pyx'],
                        libraries=['gsl', 'gslcblas'])
          ]),
          license='BSD',
          )
