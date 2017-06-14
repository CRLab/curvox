import numpy as np

def get_voxel_resolution(pc, patch_size):
    assert pc.shape[1] == 3
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
    if OLD_FIT:
        pc_center_in_voxel_grid = (15, 15, 11)

    x = create_voxel_grid_around_point_scaled(
        pc[:, 0:3],
        center,
        voxel_resolution,
        num_voxels_per_dim=patch_size,
        pc_center_in_voxel_grid=pc_center_in_voxel_grid)

    offset = np.array(center) - np.array(pc_center_in_voxel_grid) * voxel_resolution

    return x, voxel_resolution, offset