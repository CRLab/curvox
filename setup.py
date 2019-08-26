#!/usr/bin/env python

from distutils.core import setup

setup(name="curvox",
      version='1.0.0',
      description=
      """
      Python library with utilities for converting pointclouds, meshes,
      and voxel grids between ros messages, files matrices, objects
      """,
      author='David Watkins',
      author_email='davidwatkins@cs.columbia.edu',
      url='https://github.com/crlab/curvox',
      packages=['curvox'],
      scripts=['scripts/hausdorff_distance',
               'scripts/jaccard_similarity',
               'scripts/compute_depth_normals',
               'scripts/compute_tactile_normals',
               'scripts/curvox_complete',
               'scripts/viewvox',
               'scripts/curvox_binvox_to_ply',
               'scripts/capture_ros_pc',
               'scripts/find_pc_cluster',
               'scripts/filter_pcd',
               ],
      package_dir={'': 'src'},
      )
