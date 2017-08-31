import numpy as np
import binvox_rw


def get_voxel_resolution(pc, patch_size):
    """
    This function takes in a pointcloud and returns the resolution 
    of a voxel given that there will be a fixed number of voxels. 
    For example if patch_size is 40, then we are determining the 
    side length of a single voxel in meters. Sovoxel_resolution
    may end up being something like 0.01 for a 1cm^3 voxel size

    :type pc: numpy.ndarray
    :param pc: nx3 numpy array representing a pointcloud

    :type patch_size: int
    :param patch_size: int, how many voxels are there going to be.

    :rtype voxel_resolution: float

    """

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

    voxel_resolution = (1.0 * max_dim) / patch_size

    return voxel_resolution


def get_bbox_center(pc):
    """
    This function takes an nx3 pointcloud and returns a tuple
    (x,y,z) which is the center of the bbox that contains
    the pointcloud

    :type pc: numpy.ndarray
    :param pc: a nx3 numpy array representing a pointcloud
    :rtype numpy.ndarray
    """

    if not pc.shape[1] == 3:
        raise Exception("Invalid pointcloud size, should be nx3, but is {}".format(pc.shape))

    min_x = pc[:, 0].min()
    min_y = pc[:, 1].min()
    min_z = pc[:, 2].min()
    max_x = pc[:, 0].max()
    max_y = pc[:, 1].max()
    max_z = pc[:, 2].max()

    center = np.array([min_x + (max_x - min_x) / 2.0,
                       min_y + (max_y - min_y) / 2.0,
                       min_z + (max_z - min_z) / 2.0])

    return center


def voxelize_points(points, pc_bbox_center, voxel_resolution, num_voxels_per_dim, pc_center_in_voxel_grid):

    """
    This function takes a pointcloud and produces a an occupancy map or voxel grid surrounding the points.

    :type points: numpy.ndarray
    :param points: an nx3 numpy array representing a pointcloud

    :type pc_bbox_center: numpy.ndarray
    :param pc_bbox_center: numpy.ndarray of shape (3,) representing the center of the bbox that contains points

    :type voxel_resolution: float
    :param voxel_resolution: float describing in meters the length of an individual voxel edge. i.e 0.01 would
    mean each voxel is 1cm^3

    :type num_voxels_per_dim: int
    :param num_voxels_per_dim: how many voxels along a dimension. normally 40, for a 40x40x40 voxel grid

    :type pc_center_in_voxel_grid: tuple
    :param pc_center_in_voxel_grid: (x,y,z) in voxel coords of where to place the center of the points in the voxel grid
    if using 40x40x40 voxel grid, then pc_center_in_voxel_grid = (20,20,20).  We often using something more
    like (20,20,18) when doing shape completion so there is more room in the back of the grid for the
    object to be completed.
    """

    # this is the voxel grid we are going to return
    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim,
                           1), dtype=np.bool)

    # take the points and convert them from meters to voxel space coords
    centered_scaled_points = np.floor(
        (points - np.array(pc_bbox_center) + np.array(
            pc_center_in_voxel_grid) * voxel_resolution) / voxel_resolution)

    # remove any points that are beyond the area that falls in our voxel grid
    mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
    centered_scaled_points = centered_scaled_points[mask]

    # if we don't have any more points that fall within our voxel grid
    # return an empty grid
    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    # remove any points that are outside of the region we are voxelizing
    # as they are to small.
    mask = centered_scaled_points.min(axis=1) > 0
    centered_scaled_points = centered_scaled_points[mask]

    # if we don't have any more points that fall within our voxel grid,
    # return an empty grid
    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    # treat our remaining points as ints, since we are already in voxel coordinate space.
    # this points shoule be things like (5, 6, 7) which represent indices in the voxel grid.
    csp_int = centered_scaled_points.astype(int)

    # create a mask from our set of points.
    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2],
            np.zeros((csp_int.shape[0]), dtype=int))

    # apply the mask to our voxel grid setting voxel that had points in them to be occupied
    voxel_grid[mask] = 1

    return voxel_grid


def pc_to_binvox_for_shape_completion(points,
                                      patch_size):
    """
    This function creates a binvox object from a pointcloud.  The voxel grid is slightly off center from the
    pointcloud bbox center so that the back of the grid has more room for the completion.

    :type points: numpy.ndarray
    :param points: nx3 numpy array representing a pointcloud

    :type patch_size: int
    :param patch_size: how many voxels along a single dimension of the voxel grid.
    Ex: patch_size=40 gives us a 40^3 voxel grid

    :rtype: binvox_rw.Voxels

    """

    if points.shape[1] != 3:
        raise Exception("Invalid pointcloud size, should be nx3, but is {}".format(points.shape))

    # how much of the voxel grid do we want our pointcloud to fill.
    # make this < 1 so that there is some padding on the edges
    PERCENT_PATCH_SIZE = (4.0/5.0)

    # Where should the center of the points be placed inside the voxel grid.
    # normally make PERCENT_Z < 0.5 so that the points are placed towards the front of the grid
    # this leaves more room for the shape completion to fill in the occluded back half of the occupancy grid.
    PERCENT_X = 0.5
    PERCENT_Y = 0.5
    PERCENT_Z = 0.45

    # get the center of the pointcloud in meters. Ex: center = np.array([0.2, 0.1, 2.0])
    center = get_bbox_center(points)

    # get the size of an individual voxel. Ex: voxel_resolution=0.01 meaning 1cm^3 voxel
    # PERCENT_PATCH_SIZE determines how much extra padding to leave on the sides
    voxel_resolution = get_voxel_resolution(points, PERCENT_PATCH_SIZE * patch_size)

    # this tuple is where we want to stick the center of the pointcloud in our voxel grid
    # Ex: (20, 20, 18) leaving some extra room in the back half.
    pc_center_in_voxel_grid = (patch_size*PERCENT_X, patch_size*PERCENT_Y, patch_size*PERCENT_Z)

    # create a voxel grid.
    vox_np = voxelize_points(
        points=points[:, 0:3],
        pc_bbox_center=center,
        voxel_resolution=voxel_resolution,
        num_voxels_per_dim=patch_size,
        pc_center_in_voxel_grid=pc_center_in_voxel_grid)

    # location in meters of the bottom corner of the voxel grid in world space
    offset = np.array(center) - np.array(pc_center_in_voxel_grid) * voxel_resolution

    # create a voxel grid object to contain the grid, shape, offset in the world, and grid resolution
    vox = binvox_rw.Voxels(vox_np, vox_np.shape, tuple(offset), voxel_resolution * patch_size, "xyz")
    return vox