#!/usr/bin/env python
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name='cometsuite',
          version='0.1.0',
          description='Comet dust dynamics.',
          author="Michael S. P. Kelley",
          author_email="msk@astro.umd.edu",
          packages=find_packages(),
          license='BSD',
          )
