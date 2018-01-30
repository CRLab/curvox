import pcl
import plyfile
import numpy
import scipy.spatial
import mcubes
import tempfile
import meshlabxml
from pyhull.convex_hull import ConvexHull
from curvox import pc_vox_utils, binvox_conversions
import binvox_rw
import GPy as gpy
import types
import itertools

try:
    import pycuda
    useGPU = True
except ImportError as _:
    useGPU = False


# Helper functions


def smooth_ply(ply_data):
    # Store ply_data in temporary file
    tmp_ply_filename = tempfile.mktemp(suffix=".ply")
    ply_data.write(open(tmp_ply_filename, 'w'))

    # Initialize meshlabserver and meshlabxml script
    unsmoothed_mesh = meshlabxml.FilterScript(file_in=tmp_ply_filename, file_out=tmp_ply_filename, ml_version="1.3.2")
    meshlabxml.smooth.laplacian(unsmoothed_mesh, iterations=6)
    unsmoothed_mesh.run_script(print_meshlabserver_output=False, skip_error=True)

    # Read back and store new data
    ply_data_smoothed = plyfile.PlyData.read(open(tmp_ply_filename, 'r'))
    return ply_data_smoothed


def _generate_gaussian_process_points(points, offsets, observed_value, offset_value):
    offset_points = numpy.subtract(points, offsets)

    new_points = numpy.concatenate([points, offset_points])
    observations = numpy.ones((new_points.shape[0], 1))

    # Assign first half to observed_value and second half to offset_value
    observations[:points.shape[0]] = observed_value
    observations[points.shape[0]:] = offset_value

    return new_points, observations


def generate_ply_data(points, faces):
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


def read_obj_file(obj_filepath):
    with open(obj_filepath, 'r') as f:
        lines = f.readlines()
        verts = numpy.array([[float(v) for v in l.strip().split(' ')[1:]] for l in lines if
                          l[0] == 'v' and len(l.strip().split(' ')) == 4])
        faces = numpy.array([[int(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0] == 'f'])
    faces -= 1
    return verts, faces


def convert_obj_to_ply(obj_filepath, ply_filepath):
    verts, faces = read_obj_file(obj_filepath)
    ply_data = generate_ply_data(verts, faces)
    ply_data.write(open(ply_filepath, 'w'))


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

    :param points:
    :param kwargs:
    :return:
    """
    smooth = kwargs.get("smooth", False)

    # Triangulate points in (x, y) coordinate plane
    tri = scipy.spatial.Delaunay(points[:, 0:2])

    ply_data = generate_ply_data(points, tri.simplices)
    if smooth:
        ply_data = smooth_ply(ply_data)

    # Generate ply data
    return ply_data


def marching_cubes_completion(points, **kwargs):
    """

    :param points:
    :param kwargs:
    :return:
    """
    patch_size = kwargs.get("patch_size", 120)
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    marching_cubes_resolution = kwargs.get("marching_cubes_resolution", 0.5)
    smooth = kwargs.get("smooth", False)

    voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid = pc_vox_utils.pc_to_binvox(
        points=points,
        patch_size=patch_size,
        percent_offset=percent_offset,
        percent_patch_size=percent_patch_size
    )

    vertices, faces = mcubes.marching_cubes(voxel_grid.data, marching_cubes_resolution)
    vertices = pc_vox_utils.rescale_mesh(vertices, voxel_center, voxel_resolution, center_point_in_voxel_grid)

    ply_data = generate_ply_data(vertices, faces)

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

    ply_data = generate_ply_data(points, hull.vertices)

    # If we are smoothing use meshlabserver to smooth over mesh
    if smooth:
        ply_data = smooth_ply(ply_data)

    return ply_data


def _gaussian_process_helper(points, observations, **kwargs):
    patch_size = kwargs.get("patch_size", 20)
    percent_offset = kwargs.get("percent_offset", (0.5, 0.5, 0.45))
    smooth = kwargs.get("smooth", False)
    kernel_variance = kwargs.get("kernel_variance", 0.01)
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    batch_size = kwargs.get("batch_size", 1000)

    print("Here")

    # Generate gaussian process
    kernel = gpy.kern.RBF(input_dim=3, variance=kernel_variance, useGPU=useGPU)

    print("initialized kernel")

    # Create voxel grid to use for gaussian process
    voxel_grid, voxel_center, voxel_resolution, center_point_in_voxel_grid = \
        pc_vox_utils.pc_to_binvox(points, patch_size=patch_size, percent_offset=percent_offset, percent_patch_size=percent_patch_size)

    print("Created voxel grid: {}".format(voxel_grid.data.shape))

    # Where am I putting my point cloud relative to the center of my voxel grid
    # ex. (20, 20, 20) or (20, 20, 18)
    pc_center_in_voxel_grid = (patch_size * percent_offset[0], patch_size * percent_offset[1], patch_size * percent_offset[2])

    # Find all positions that are known to be empty
    ternary_voxel_grid = pc_vox_utils.get_ternary_voxel_grid(voxel_grid)
    unoccupied_voxels = numpy.zeros(voxel_grid.data.shape)
    unoccupied_voxels[ternary_voxel_grid == 0] = 1

    # Get indices of every point marked as empty
    offset = numpy.array(voxel_center) - numpy.array(pc_center_in_voxel_grid) * voxel_resolution
    vox = binvox_rw.Voxels(unoccupied_voxels, unoccupied_voxels.shape, tuple(offset), voxel_resolution * patch_size, "xyz")
    unoccupied_points = binvox_conversions.binvox_to_pcl(vox)
    unoccupied_points = unoccupied_points[0:unoccupied_points.size:10]

    print("Unoccupied points: {}".format(unoccupied_points.shape))

    unoccupied_observations = numpy.zeros((unoccupied_points.shape[0], 1))

    points = numpy.concatenate([points, unoccupied_points], axis=0)
    observations = numpy.concatenate([observations, unoccupied_observations], axis=0)

    while points.shape[0] > batch_size:
        points = points[::2, :]
        observations = observations[::2, :]
        print(points.shape, observations.shape)

    gp = gpy.models.GPRegression(points, observations, kernel, noise_var=0.005)
    gp.optimize()

    # Generate query points
    xs = numpy.arange(voxel_center[0] - 0.5 * voxel_resolution * patch_size,
                      voxel_center[0] + 0.5 * voxel_resolution * patch_size, voxel_resolution)
    ys = numpy.arange(voxel_center[1] - 0.5 * voxel_resolution * patch_size,
                      voxel_center[1] + 0.5 * voxel_resolution * patch_size, voxel_resolution)
    zs = numpy.arange(voxel_center[2] - 0.5 * voxel_resolution * patch_size,
                      voxel_center[2] + 0.5 * voxel_resolution * patch_size, voxel_resolution)

    prediction_points = numpy.vstack(numpy.meshgrid(xs, ys, zs)).reshape(3, -1).T

    prediction, _ = gp.predict(prediction_points, full_cov=True)

    output_grid = numpy.zeros((patch_size, patch_size, patch_size))
    for counter, (i, j, k) in enumerate(itertools.product(range(patch_size), range(patch_size), range(patch_size))):
        output_grid[i][j][k] = prediction[counter]

    vertices, faces = mcubes.marching_cubes(output_grid[:, :, :], 0.5)
    vertices = pc_vox_utils.rescale_mesh(vertices, voxel_center, voxel_resolution, pc_center_in_voxel_grid)

    ply_data = generate_ply_data(vertices, faces)

    # If we are smoothing use meshlabserver to smooth over mesh
    if smooth:
        ply_data = smooth_ply(ply_data)

    return ply_data


def gaussian_process_completion(points, **kwargs):
    print("Entering gaussian_process_completion")
    print("Generating observation points")
    points, observations = \
        _generate_gaussian_process_points(points, offsets=(0, 0, 0.01), observed_value=1, offset_value=-1)
    print(points.shape, observations.shape)

    return _gaussian_process_helper(points, observations, **kwargs)


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


def gaussian_process_tactile_completion(depth_points, tactile_points, **kwargs):
    # Copy depth points
    depth_points, depth_observations = \
        _generate_gaussian_process_points(depth_points, offsets=(0, 0, 0.005), observed_value=1, offset_value=0)

    # Copy tactile points
    tactile_points, tactile_observations = \
        _generate_gaussian_process_points(tactile_points, offsets=(0, 0, -0.005), observed_value=0, offset_value=1)

    points = numpy.concatenate([depth_points, tactile_points], axis=0)
    observations = numpy.concatenate([depth_observations, tactile_observations], axis=0)

    return _gaussian_process_helper(points, observations, **kwargs)


COMPLETION_METHODS = [
    delaunay_completion,
    marching_cubes_completion,
    qhull_completion,
    gaussian_process_completion
]
TACTILE_COMPLETION_METHODS = [delaunay_tactile_completion,
                              marching_cubes_tactile_completion,
                              qhull_tactile_completion,
                              gaussian_process_tactile_completion]
