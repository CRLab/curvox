import numpy as np
import binvox_rw
import numba
import mcubes
import cloud_to_mesh_conversions


@numba.jit
def get_voxel_resolution(pc, patch_size):
    """
    This function takes in a pointcloud and returns the resolution 
    of a voxel given that there will be a fixed number of voxels. 
    For example if patch_size is 40, then we are determining the 
    side length of a single voxel in meters. Sovoxel_resolution
    may end up being something like 0.01 for a 1cm^3 voxel size
    jaccard_distance
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


@numba.jit
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


@numba.jit
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
                           num_voxels_per_dim), dtype=np.bool)

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
    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2])

    # apply the mask to our voxel grid setting voxel that had points in them to be occupied
    voxel_grid[mask] = 1

    return voxel_grid


def pc_to_binvox(points, **kwargs):
    """
    This function creates a binvox object from a pointcloud. The voxel grid is slightly off center from the
    pointcloud bbox center so that the back of the grid has more room for the completion.

    :type points: numpy.ndarray
    :param points: nx3 numpy array representing a pointcloud

    :rtype: binvox_rw.Voxels

    :param kwargs:
        See below

    :Keyword Arguments:
        * *patch_size* (``int``) --
            how many voxels along a single dimension of the voxel grid.
            Ex: patch_size=40 gives us a 40^3 voxel grid
            Defaults to 40
        * *percent_patch_size* (``float``) --
            how much of the voxel grid do we want our pointcloud to fill.
            make this < 1 so that there is some padding on the edges
            Defaults to 0.8
        * *percent_offset* (``tuple``) --
            Where should the center of the points be placed inside the voxel grid.
            normally make PERCENT_Z < 0.5 so that the points are placed towards the front of the grid
            this leaves more room for the shape completion to fill in the occluded back half of the occupancy grid.
    """
    patch_size = kwargs.get("patch_size", 40)
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)

    if points.shape[1] != 3:
        raise Exception("Invalid pointcloud size, should be nx3, but is {}".format(points.shape))

    if len(percent_offset) != 3:
        raise Exception("Percent offset should be a tuple of size 3, instead got {}".format(percent_offset))

    percent_x, percent_y, percent_z = percent_offset

    # get the center of the pointcloud in meters. Ex: center = np.array([0.2, 0.1, 2.0])
    voxel_center = get_bbox_center(points)

    # get the size of an individual voxel. Ex: voxel_resolution=0.01 meaning 1cm^3 voxel
    # PERCENT_PATCH_SIZE determines how much extra padding to leave on the sides
    voxel_resolution = get_voxel_resolution(points, percent_patch_size * patch_size)

    # this tuple is where we want to stick the center of the pointcloud in our voxel grid
    # Ex: (20, 20, 18) leaving some extra room in the back half.
    pc_center_in_voxel_grid = (patch_size*percent_x, patch_size*percent_y, patch_size*percent_z)

    # create a voxel grid.
    vox_np = voxelize_points(
        points=points[:, 0:3],
        pc_bbox_center=voxel_center,
        voxel_resolution=voxel_resolution,
        num_voxels_per_dim=patch_size,
        pc_center_in_voxel_grid=pc_center_in_voxel_grid)

    # location in meters of the bottom corner of the voxel grid in world space
    offset = np.array(voxel_center) - np.array(pc_center_in_voxel_grid) * voxel_resolution

    # create a voxel grid object to contain the grid, shape, offset in the world, and grid resolution
    voxel_grid = binvox_rw.Voxels(vox_np, vox_np.shape, tuple(offset), voxel_resolution * patch_size, "xyz")

    # Where am I putting my point cloud relative to the center of my voxel grid
    # ex. (20, 20, 20) or (20, 20, 18)
    center_point_in_voxel_grid = (patch_size * percent_x, patch_size * percent_y, patch_size * percent_z)

    return voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid


@numba.jit
def get_ternary_voxel_grid(binary_voxel_grid):
    """
    Takes a binary occupancy voxel grid for the surface of the object and
    returns a ternary occupancy voxel grid.
    :param binary_voxel_grid: a voxel grid that indicates whether a voxel is
    occupied by the visible surface ("1") or not occupied by the visible
    surface ("0"). If you're seeing a box, the "1"s would represent the location
    of the part of the box's surface that you can see, while the "0" would
    represent everything else.
    :param method: Can be 'simple' or 'projection'.
    :return: a voxel grid that indicates whether a voxel is visually occluded
    ("2"), occupied by the visible surface ("1"), or visibly known to be
    unoccupied ("0").
    """

    if isinstance(binary_voxel_grid, binvox_rw.Voxels):
        binary_voxel_grid = binary_voxel_grid.data

    if not isinstance(binary_voxel_grid, np.ndarray):
        raise ValueError("binary_voxel_grid must be Voxels or ndarray")

    voxel_grid_shape = binary_voxel_grid.shape
    assert len(voxel_grid_shape) == 3


    # Initialize all ternary grid values to 0.
    ternary_voxel_grid = np.zeros(voxel_grid_shape)
    # The 'simple' method assumes that the camera is an infinite distance
    # away from the object and thus considers as occluded every z value
    # behind the surface for a fixed x and y. Perspective isn't taken into
    # account.

    for i in range(voxel_grid_shape[0]):
        for j in range(voxel_grid_shape[1]):
            for k in range(voxel_grid_shape[2]):
                if binary_voxel_grid[i, j, k] > 0:
                    # Surface found. set surface to 1 in the ternary_voxel
                    # grid, and everything behind it to 2.
                    ternary_voxel_grid[i, j, k] = 1
                    ternary_voxel_grid[i, j, k + 1:voxel_grid_shape[2]] = 2
                    break
    return ternary_voxel_grid


@numba.jit
def rescale_mesh(vertices, patch_center, voxel_resolution, pc_center_in_voxel_grid):
    return vertices * voxel_resolution - np.array(pc_center_in_voxel_grid) * voxel_resolution + np.array(patch_center)


@numba.jit
def create_voxel_grid_around_point_scaled(
        points,
        patch_center,
        voxel_resolution,
        num_voxels_per_dim,
        pc_center_in_voxel_grid
):
    voxel_grid = np.zeros((num_voxels_per_dim, num_voxels_per_dim, num_voxels_per_dim, 1), dtype=np.float32)

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


def binvox_to_ply(voxel_grid, **kwargs):
    """

    :param voxel_grid:
    :type voxel_grid: binvox_rw.Voxels
    :param kwargs:
    :return:
    """
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    marching_cubes_resolution = kwargs.get("marching_cubes_resolution", 0.5)

    patch_size = voxel_grid.dims[0]
    pc_center_in_voxel_grid = (patch_size * percent_offset[0], patch_size * percent_offset[1], patch_size * percent_offset[2])
    voxel_resolution = voxel_grid.scale / patch_size
    center_point_in_voxel_grid = voxel_grid.translate + np.array(pc_center_in_voxel_grid) * voxel_resolution

    vertices, faces = mcubes.marching_cubes(voxel_grid.data, marching_cubes_resolution)
    vertices = vertices * voxel_resolution - np.array(pc_center_in_voxel_grid) * voxel_resolution + np.array(center_point_in_voxel_grid)

    ply_data = cloud_to_mesh_conversions.generate_ply_data(vertices, faces)

    # Export to plyfile type
    return ply_data

