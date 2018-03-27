import curvox.binvox_conversions
import curvox.pc_vox_utils
import curvox.cloud_transformations

import tf_conversions
import pcl
import PyKDL
import numpy as np
import binvox_rw
import math
import os
import numba


def generate_tactile_points(gt_binvox, mesh_pose, patch_size, tactile_sample_count=40):

    # Load in ground truth point cloud and metadata
    gt_pc = curvox.binvox_conversions.binvox_to_pcl(gt_binvox)

    # Remap partial_pc and gt_pc to camera frame
    gt_pc_cf = curvox.cloud_transformations.transform_cloud(gt_pc, mesh_pose)

    # first we are voxelizing the gt independent of the partial, this
    # will make it easier to get tactile sensor observations
    gt_binvox_cf, gt_voxel_center, gt_voxel_resolution, center_point_in_voxel_grid = curvox.pc_vox_utils.pc_to_binvox(gt_pc_cf)
    gt_binvox_cf = gt_binvox_cf.data

    # Now we look for the voxels in which tactile contact occurs.
    xs = np.random.random_integers(0, patch_size - 1, size=tactile_sample_count)
    ys = np.random.random_integers(0, patch_size - 1, size=tactile_sample_count)

    contact_voxels = []
    for x, y in zip(xs, ys):
        for z in xrange(patch_size):
            if gt_binvox_cf[x, y, patch_size - z - 1] == 1:
                contact_voxels.append([x, y, patch_size - z - 1])
                break

    # these are voxel indices in the gt_cf_voxelized grid
    contact_voxels = np.array(contact_voxels)

    if len(contact_voxels) > 0:
        # Now we convert contact voxels back to camera_frame
        # this cloud is in meters in camera frame, x,y,z are real values
        contact_points_cf = contact_voxels * gt_voxel_resolution - np.array(center_point_in_voxel_grid) * gt_voxel_resolution + gt_voxel_center
    else:
        contact_points_cf = np.array([])

    contact_points_cf = contact_points_cf.astype(np.float32)
    return contact_points_cf
