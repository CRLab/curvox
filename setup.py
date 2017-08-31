#!/usr/bin/env python

from distutils.core import setup

setup(name="curvox",
      version='1.0',
      description=
      """
      Python library with utilities for converting pointclouds, meshes,
      and voxel grids between ros messages, files matrices, objects
      """,
      author='David Watkins',
      author_email='davidwatkins@cs.columbia.edu',
      url='https://github.com/CURG/curvox',
      packages=['curvox'],
      dependency_links=['https://github.com/CURG/binvox-rw-py/tarball/master#egg=binvox-rw-1.0'],
      install_requires=[
          "numpy >= 1.8",
          "binvox-rw >= 1.0",
          "plyfile >= 0.4"
      ])
