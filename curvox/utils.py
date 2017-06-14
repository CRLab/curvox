import numpy as np
import binvox_rw

PERCENT_PATCH_SIZE = (4.0/5.0)
PERCENT_X = 0.5
PERCENT_Y = 0.5
PERCENT_Z = 0.45

def pcd2binvox(pcd_filepath, binvox_filepath, grid_dim):
    '''
    This function takes a pcd filepath, and converts it to
    a binvox file.  This does not voxeliz the point cloud,
    it assumes each point can be represented as indices
    ex cloud:
    pts = [(0,1,0), (0,10,23), ...]
    would be converted to a binvox with voxels 
    (0,1,0) and (0 , 10, 23) occupied
    '''

    pc = pcl.load(pcd_filepath)
    pc_arr = pc.to_array().astype(int)
    pc_mask = (pc_arr[:,0], pc_arr[:,1], pc_arr[:,2])

    out = np.zeros((grid_dim, grid_dim, grid_dim))
    out[pc_mask] = 1
    out_int = out.astype(int)

    bv = binvox_rw.Voxels(out_int, out_int.shape, (0,0,0),1, 'xyz')
    binvox_rw.write(bv, open(binvox_filepath, 'w'))


def get_voxel_resolution(pc, patch_size):
    '''
    This function takes in a pointcloud and returns the resolution 
    of a voxel given that there will be a fixed number of voxels. 
    For example if patch_size is 40, then we are determining the 
    side length of a single voxel in meters. 
    '''

    if not pc.shape[1] == 3:
        raise Exception("Invalid pointcloud size, should be nx3, but is {}".format(pc.shape))

    min_x = pc[:, 0].min()
    min_y = pc[:, 1].min()
    min_z = pc[:, 2].min()
    max_x = pc[:, 0].max()
    max_y = pc[:, 1].max()
    max_z = pc[:, 2].max()

    max_dim = max((max_x - min_x),
                  (max_y - min_y),
                  (max_z - min_z))

    print max_dim
    print PERCENT_PATCH_SIZE
    print patch_size

    voxel_resolution = (1.0*max_dim) / (PERCENT_PATCH_SIZE * patch_size)
    return voxel_resolution


def get_center(pc):

    assert pc.shape[1] == 3
    min_x = pc[:, 0].min()
    min_y = pc[:, 1].min()
    min_z = pc[:, 2].min()
    max_x = pc[:, 0].max()
    max_y = pc[:, 1].max()
    max_z = pc[:, 2].max()

    center = (min_x + (max_x - min_x) / 2.0,
              min_y + (max_y - min_y) / 2.0,
              min_z + (max_z - min_z) / 2.0)

    return center


def create_voxel_grid_around_point_scaled(points, patch_center,
                                          voxel_resolution, num_voxels_per_dim,
                                          pc_center_in_voxel_grid):
    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim,
                           1), dtype=np.float32)

    centered_scaled_points = np.floor(
        (points - np.array(patch_center) + np.array(
            pc_center_in_voxel_grid) * voxel_resolution) / voxel_resolution)

    mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
    centered_scaled_points = centered_scaled_points[mask]

    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    mask = centered_scaled_points.min(axis=1) > 0
    centered_scaled_points = centered_scaled_points[mask]

    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    csp_int = centered_scaled_points.astype(int)

    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2],
            np.zeros((csp_int.shape[0]), dtype=int))

    voxel_grid[mask] = 1

    return voxel_grid

def build_test_from_pc_scaled(pc,
                              patch_size,
                              custom_scale=1,
                              custom_offset=(0, 0, 0)):

    center = get_center(pc)
    voxel_resolution = get_voxel_resolution(pc, patch_size)

    pc_center_in_voxel_grid = (patch_size*PERCENT_X, patch_size*PERCENT_Y, patch_size*PERCENT_Z)

    x = create_voxel_grid_around_point_scaled(
        pc[:, 0:3],
        center,
        voxel_resolution,
        num_voxels_per_dim=patch_size,
        pc_center_in_voxel_grid=pc_center_in_voxel_grid)

    offset = np.array(center) - np.array(pc_center_in_voxel_grid) * voxel_resolution

    return x, voxel_resolution, offset
    