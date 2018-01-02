import pcl
import plyfile
import numpy
import scipy.spatial
import mcubes
import tempfile
import meshlabxml
from pyhull.convex_hull import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from curvox import pc_vox_utils, binvox_conversions
import gpflow
import binvox_rw
import types
import itertools


# Helper functions


def smooth_ply(ply_data):
    # Store ply_data in temporary file
    tmp_ply_fp, tmp_ply_filename = tempfile.mktemp()
    ply_data.write(tmp_ply_fp)

    # Initialize meshlabserver and meshlabxml script
    unsmoothed_mesh = meshlabxml.FilterScript(file_in=tmp_ply_filename, file_out=tmp_ply_filename, ml_version="1.3.2")
    meshlabxml.smooth.laplacian(unsmoothed_mesh, iterations=6)
    unsmoothed_mesh.run_script()

    # Read back and store new data
    ply_data_smoothed = plyfile.PlyData()
    ply_data_smoothed.read(open(tmp_ply_filename, 'r'))
    return ply_data_smoothed


def _generate_gaussian_process_points(points, offsets, observed_value, offset_value):
    offset_points = numpy.zeros(points.shape)
    offset_points = numpy.subtract(offset_points, offsets, axis=1)

    new_points = numpy.concatenate([points, offset_points])
    observations = numpy.ones((new_points.shape[0], 1))

    # Assign first half to observed_value and second half to offset_value
    observations[:points.shape[0]] = observed_value
    observations[points.shape[0]:] = offset_value

    return new_points


def _generate_prediction_points(points, patch_size, percent_offset):
    voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid = pc_vox_utils.pc_to_binvox(
        points,
        patch_size=patch_size,
        percent_offset=percent_offset
    )

    # Generate query points
    xs = numpy.arange(voxel_center[0] - 0.5 * voxel_resolution * patch_size,
                      voxel_center[0] + 0.5 * voxel_resolution * patch_size, voxel_resolution)
    ys = numpy.arange(voxel_center[1] - 0.5 * voxel_resolution * patch_size,
                      voxel_center[1] + 0.5 * voxel_resolution * patch_size, voxel_resolution)
    zs = numpy.arange(voxel_center[2] - 0.5 * voxel_resolution * patch_size,
                      voxel_center[2] + 0.5 * voxel_resolution * patch_size, voxel_resolution)

    # Stack all the points together into 3D grid
    return numpy.vstack(numpy.meshgrid(xs, ys, zs)).reshape(3, -1).T, voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid


def _generate_ply_data(points, faces):
    """

    :param points:
    :param faces:
    :return:
    """
    vertices = [(point[0], point[1], point[2]) for point in points]
    faces = [(point,) for point in faces]

    vertices_np = numpy.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_np = numpy.array(faces, dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = plyfile.PlyElement.describe(vertices_np, 'vertex')
    face_element = plyfile.PlyElement.describe(faces_np, 'face')

    return plyfile.PlyData([vertex_element, face_element], text=True)


def plot_values(points, values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    c = values[:, 0]

    ax.scatter(x, y, z, c=c, cmap=plt.hot())
    plt.show()


# Functions for saving and loading PCL and PLY files


def complete_pcd_file_and_save(pcd_filename, completion_method, **kwargs):
    """

    :param pcd_filename:
    :param completion_method:
    :type completion_method: types.FunctionType
    :param kwargs:
    :return:
    """
    if completion_method not in COMPLETION_METHODS:
        raise ValueError("completion_method must be a completion method defined in this module")

    suffix = kwargs.get("suffix", "")
    cloud = pcl.load(pcd_filename)
    points = cloud.to_array()

    plydata = completion_method(points, **kwargs)

    ply_filename = pcd_filename.replace(".pcd", suffix + ".ply")
    plydata.write(open(ply_filename, 'wb'))


def complete_tactile_pcd_file_and_save(depth_pcd_filename, tactile_pcd_filename, completion_method, **kwargs):
    """

    :param depth_pcd_filename:
    :param tactile_pcd_filename:
    :param completion_method:
    :param kwargs:
    :return:
    """
    suffix = kwargs.get("suffix", "")
    depth_cloud = pcl.load(depth_pcd_filename)
    tactile_cloud = pcl.load(tactile_pcd_filename)
    depth_points = depth_cloud.to_array()
    tactile_points = tactile_cloud.to_array()

    if completion_method not in TACTILE_COMPLETION_METHODS:
        raise ValueError("completion_method must be a tactile completion method defined in this module")

    plydata = completion_method(depth_points, tactile_points, **kwargs)

    ply_filename = depth_pcd_filename.replace(".pcd", suffix + ".ply")
    plydata.write(open(ply_filename, 'wb'))


# Depth only completions


def delaunay_completion(points, **kwargs):
    """

    :param cloud:
    :type cloud: pcl.PointCloud
    :param kwargs: Ignored
    :return:
    """
    smooth = kwargs.get("smooth", False)

    # Triangulate points in (x, y) coordinate plane
    tri = scipy.spatial.Delaunay(points[:, 0:2])

    ply_data = _generate_ply_data(points, tri.simplices)
    if smooth:
        ply_data = smooth_ply(ply_data)

    # Generate ply data
    return ply_data


def marching_cubes_completion(points, **kwargs):
    """

    :param cloud:
    :param kwargs:
    :return:
    """
    patch_size = kwargs.get("patch_size", 120)
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    marching_cubes_resolution = kwargs.get("marching_cubes_resolution", 0.5)
    smooth = kwargs.get("smooth", False)

    voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid = pc_vox_utils.pc_to_binvox(
        points,
        patch_size=patch_size,
        percent_offset=percent_offset,
        percent_patch_size=percent_patch_size
    )

    vertices, faces = mcubes.marching_cubes(voxel_grid[:, :, :, 0], marching_cubes_resolution)
    vertices = pc_vox_utils.rescale_mesh(vertices, voxel_center, voxel_resolution, center_point_in_voxel_grid)

    ply_data = _generate_ply_data(vertices, faces)

    # If we are smoothing use meshlabserver to smooth over mesh
    if smooth:
        ply_data = smooth_ply(ply_data)

    # Export to plyfile type
    return ply_data


def qhull_completion(points, **kwargs):
    smooth = kwargs.get("smooth", False)

    hull = ConvexHull(points)

    # Fix inverted normals from pyhull
    hull.vertices = [vertex[::-1] for vertex in hull.vertices]

    ply_data = _generate_ply_data(points, hull.vertices)

    # If we are smoothing use meshlabserver to smooth over mesh
    if smooth:
        ply_data = smooth_ply(ply_data)

    return ply_data


def gaussian_process_completion(points, **kwargs):
    patch_size = kwargs.get("patch_size", 120)
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    smooth = kwargs.get("smooth", False)
    kernel_variance = kwargs.get("kernel_variance", 0.01)

    # Generate gaussian process
    kernel = gpflow.kernels.RBF(input_dim=3, variance=kernel_variance)
    points, values = _generate_gaussian_process_points(points, (0, 0, -0.005), 1, 0)

    ternary_voxel_grid = pc_vox_utils.get_ternary_voxel_grid(voxel_grid)
    unoccupied_voxels = numpy.zeros(voxel_grid.shape)
    unoccupied_voxels[ternary_voxel_grid == 0] = 1

    # Get indices of every point marked as empty
    offset = numpy.array(center_point) - numpy.array(pc_center_in_voxel_grid) * voxel_resolution
    vox = binvox_rw.Voxels(unoccupied_voxels, unoccupied_voxels.shape, tuple(offset), voxel_resolution * patch_size,
                           "xyz")
    unoccupied_points = binvox_conversions.binvox_to_pcl(vox)
    unoccupied_points = unoccupied_points[0:unoccupied_points.size:10]

    gp = gpflow.models.GPR(points, values, kernel, noise_var=0.005)
    gp.compile()

    # Create voxel grid to use for gaussian process
    prediction_points, voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid = _generate_prediction_points(points, patch_size, percent_offset)

    prediction = gp.predict_f_full_cov(prediction_points)

    output_grid = numpy.zeros((patch_size, patch_size, patch_size))
    for counter, (i, j, k) in enumerate(itertools.product(range(patch_size), range(patch_size), range(patch_size))):
        output_grid[i][j][k] = prediction[0][counter]

    vertices, faces = mcubes.marching_cubes(output_grid[:, :, :], 0.5)
    vertices = pc_vox_utils.rescale_mesh(vertices, voxel_center, voxel_resolution, center_point_in_voxel_grid)

    ply_data = _generate_ply_data(vertices, faces)

    # If we are smoothing use meshlabserver to smooth over mesh
    if smooth:
        ply_data = smooth_ply(ply_data)

    return ply_data


# Tactile completion methods


def delaunay_tactile_completion(depth_points, tactile_points, **kwargs):
    points = numpy.concatenate([depth_points, tactile_points])
    return delaunay_completion(points, **kwargs)


def marching_cubes_tactile_completion(depth_points, tactile_points, **kwargs):
    points = numpy.concatenate([depth_points, tactile_points])
    return marching_cubes_completion(points, **kwargs)


def qhull_tactile_completion(depth_points, tactile_points, **kwargs):
    points = numpy.concatenate([depth_points, tactile_points])
    return qhull_completion(points, **kwargs)


def gaussian_process_tactile_completion(points, **kwargs):
    patch_size = kwargs.get("patch_size", 120)
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    smooth = kwargs.get("smooth", False)
    kernel_variance = kwargs.get("kernel_variance", 0.01)

    # Generate gaussian process
    kernel = gpflow.kernels.RBF(input_dim=3, variance=kernel_variance)
    points, values = _generate_gaussian_process_points(points, (0, 0, -0.005), 1, 0)

    gp = gpflow.models.GPR(points, values, kernel, noise_var=0.005)
    gp.compile()

    # Create voxel grid to use for gaussian process
    prediction_points, _, voxel_center, voxel_resolution, center_point_in_voxel_grid = _generate_prediction_points(points, patch_size, percent_offset)

    prediction = gp.predict_f_full_cov(prediction_points)

    output_grid = numpy.zeros((patch_size, patch_size, patch_size))
    for counter, (i, j, k) in enumerate(itertools.product(range(patch_size), range(patch_size), range(patch_size))):
        output_grid[i][j][k] = prediction[0][counter]

    vertices, faces = mcubes.marching_cubes(output_grid[:, :, :], 0.5)
    vertices = pc_vox_utils.rescale_mesh(vertices, voxel_center, voxel_resolution, center_point_in_voxel_grid)

    ply_data = _generate_ply_data(vertices, faces)

    # If we are smoothing use meshlabserver to smooth over mesh
    if smooth:
        ply_data = smooth_ply(ply_data)

    return ply_data


def _gaussian_process_pointcloud_depth_tactile_completion(depth_cloud, kernel_variance, tactile_cloud, patch_size, percent_x, percent_y, percent_z, percent_patch_size):
    depth_points = depth_cloud.to_array()
    tactile_points = tactile_cloud.to_array()


    # Copy depth points
    points_positive = numpy.copy(depth_points)
    # points_positive[:, 2] += 0.01
    points_negative = numpy.copy(depth_points)
    points_negative[:, 2] -= 0.005
    depth_points = numpy.concatenate([points_negative, points_positive], axis=0)
    sdf_meas_depth = numpy.ones((depth_points.shape[0], 1))
    for i in range(depth_points.shape[0]/2):
        sdf_meas_depth[i] *= 0

    out = _complete_pointcloud_partial(depth_points, patch_size, percent_x, percent_y, percent_z, percent_patch_size)
    out.write(open("depth.ply", 'w'))

    # Copy tactile points
    points_positive = numpy.copy(tactile_points)
    # points_positive[:, 2] += 0.01
    points_negative = numpy.copy(tactile_points)
    points_negative[:, 2] -= 0.005
    tactile_points = numpy.concatenate([points_positive, points_negative], axis=0)
    sdf_meas_tactile = numpy.ones((tactile_points.shape[0], 1))
    for i in range(tactile_points.shape[0]/2):
        sdf_meas_tactile[i] *= 0

    out = _complete_pointcloud_partial(tactile_points, patch_size, percent_x, percent_y, percent_z, percent_patch_size)
    out.write(open("tactile.ply", 'w'))

    points = numpy.concatenate([depth_points, tactile_points], axis=0)
    sdf_meas = numpy.concatenate([sdf_meas_depth, sdf_meas_tactile], axis=0)

    # Generate gaussian process
    kernel = gpflow.kernels.RBF(input_dim=3, variance=kernel_variance)

    # Create voxel grid to use for gaussian process
    center_point = pc_vox_utils.get_bbox_center(points)
    voxel_resolution = pc_vox_utils.get_voxel_resolution(points, patch_size)

    # Where am I putting my point cloud relative to the center of my voxel grid
    # ex. (20, 20, 20) or (20, 20, 18)
    pc_center_in_voxel_grid = (patch_size * percent_x, patch_size * percent_y, patch_size * percent_z)

    voxel_grid = pc_vox_utils.voxelize_points(points, center_point, voxel_resolution, patch_size,
                                              pc_center_in_voxel_grid)

    ternary_voxel_grid = pc_vox_utils.get_ternary_voxel_grid(voxel_grid)
    unoccupied_voxels = numpy.zeros(voxel_grid.shape)
    unoccupied_voxels[ternary_voxel_grid == 0] = 1

    # Get indices of every point marked as empty
    offset = numpy.array(center_point) - numpy.array(pc_center_in_voxel_grid) * voxel_resolution
    vox = binvox_rw.Voxels(unoccupied_voxels, unoccupied_voxels.shape, tuple(offset), voxel_resolution * patch_size, "xyz")
    unoccupied_points = binvox_conversions.binvox_to_pcl(vox)
    unoccupied_points = unoccupied_points[0:unoccupied_points.size:10]

    sdf_meas_unoccupied_points = numpy.zeros((unoccupied_points.shape[0], 1))

    points = numpy.concatenate([points, unoccupied_points], axis=0)
    sdf_meas = numpy.concatenate([sdf_meas, sdf_meas_unoccupied_points], axis=0)

    gp = gpflow.models.GPR(points, sdf_meas, kernel, noise_var=0.005)
    gp.compile()

    # Generate query points
    xs = numpy.arange(center_point[0] - 0.5 * voxel_resolution * patch_size,
                      center_point[0] + 0.5 * voxel_resolution * patch_size, voxel_resolution)
    ys = numpy.arange(center_point[1] - 0.5 * voxel_resolution * patch_size,
                      center_point[1] + 0.5 * voxel_resolution * patch_size, voxel_resolution)
    zs = numpy.arange(center_point[2] - 0.5 * voxel_resolution * patch_size,
                      center_point[2] + 0.5 * voxel_resolution * patch_size, voxel_resolution)

    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    points = numpy.array(points)
    prediction = gp.predict_f_full_cov(points)

    output_grid = numpy.zeros((patch_size, patch_size, patch_size))
    counter = 0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                output_grid[i][j][k] = prediction[0][counter]
                counter += 1

    vertices, faces = mcubes.marching_cubes(output_grid[:, :, :], 0.5)
    vertices = rescale_mesh(vertices, center_point, voxel_resolution, pc_center_in_voxel_grid)

    # Export to plyfile type
    return _generate_ply_data(vertices, faces)


def gaussian_process_depth_tactile_completion(depth_pcd_filenames, tactile_pcd_filenames, **kwargs):
    patch_size = kwargs.get("patch_size", 120)
    percent_x = kwargs.get("percent_x", 0.5)
    percent_y = kwargs.get("percent_y", 0.5)
    percent_z = kwargs.get("percent_z", 0.45)
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    suffix = kwargs.get("suffix", "")

    for (depth_pcd_filename, tactile_pcd_filename) in zip(depth_pcd_filenames, tactile_pcd_filenames):
        depth_cloud = pcl.load(depth_pcd_filename)
        tactile_cloud = pcl.load(tactile_pcd_filename)

        plydata = _gaussian_process_pointcloud_depth_tactile_completion(depth_cloud, tactile_cloud, patch_size, percent_x, percent_y, percent_z, percent_patch_size)

        ply_filename = depth_pcd_filename.replace(".pcd", suffix + ".ply")
        plydata.write(open(ply_filename, 'wb'))


COMPLETION_METHODS = [delaunay_completion, marching_cubes_completion, qhull_completion, gaussian_process_completion]
TACTILE_COMPLETION_METHODS = []
