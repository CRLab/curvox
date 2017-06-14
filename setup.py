#!/usr/bin/env python

from distutils.core import setup

setup(name="Curvox",
	  version='1.0',
	  description='Python library for creating and initializing point clouds from voxel grids',
	  author='David Watkins',
	  author_email='davidwatkins@cs.columbia.edu',
	  url='https://github.com/CURG/curvox',
	  packages=['curvox'],
	  install_requires=[
	  	"numpy >= 1.12.1"
	  ])