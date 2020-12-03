import pypcd
import plyfile
import numpy
import scipy.spatial
import mcubes
import tempfile
import meshlabxml
from pyhull.convex_hull import ConvexHull
from curvox import pc_vox_utils
import types


# Binvox functions


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
    center_point_in_voxel_grid = voxel_grid.translate + numpy.array(pc_center_in_voxel_grid) * voxel_resolution

    vertices, faces = mcubes.marching_cubes(voxel_grid.data, marching_cubes_resolution)
    vertices = vertices * voxel_resolution - numpy.array(pc_center_in_voxel_grid) * voxel_resolution + numpy.array(center_point_in_voxel_grid)

    ply_data = generate_ply_data(vertices, faces)

    # Export to plyfile type
    return ply_data


def binvox_to_ply(voxel_grid, **kwargs):
    """

    :param voxel_grid:
    :type voxel_grid: numpy.Array
    :param kwargs:
    :return:
    """
    marching_cubes_resolution = kwargs.get("marching_cubes_resolution", 0.5)

    patch_size = voxel_grid.shape[0]
    vertices, faces = mcubes.marching_cubes(voxel_grid, marching_cubes_resolution)
    vertices = vertices * voxel_resolution

    ply_data = generate_ply_data(vertices, faces)

    # Export to plyfile type
    return ply_data


# Helper functions


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


def smooth_ply(ply_data):
    # Store ply_data in temporary file
    tmp_ply_filename = tempfile.mktemp(suffix=".ply")
    with open(tmp_ply_filename, 'w') as tmp_ply_file:
        ply_data.write(tmp_ply_file)

    # Initialize meshlabserver and meshlabxml script
    unsmoothed_mesh = meshlabxml.FilterScript(file_in=tmp_ply_filename, file_out=tmp_ply_filename, ml_version="1.3.2")
    meshlabxml.smooth.laplacian(unsmoothed_mesh, iterations=6)
    unsmoothed_mesh.run_script(print_meshlabserver_output=False, skip_error=True)

    # Read back and store new data
    with open(tmp_ply_filename, 'r') as tmp_ply_file:
        ply_data_smoothed = plyfile.PlyData.read(tmp_ply_file)
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


def convert_obj_to_ply(obj_filepath):
    verts, faces = read_obj_file(obj_filepath)
    ply_data = generate_ply_data(verts, faces)
    return ply_data


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
    cloud = pypcd.Pointcloud.from_file(pcd_filename)
    points = curvox.cloud_conversions.pcl_to_np(cloud)

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
    depth_cloud = pypcd.Pointcloud.from_file(depth_pcd_filename)
    depth_points = curvox.cloud_conversions.pcl_to_np(depth_cloud)
    tactile_cloud = pypcd.Pointcloud.from_file(tactile_pcd_filename)
    tactile_points = curvox.cloud_conversions.pcl_to_np(tactile_cloud)

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


COMPLETION_METHODS = [
    delaunay_completion,
    marching_cubes_completion,
    qhull_completion,
]
TACTILE_COMPLETION_METHODS = [
    delaunay_tactile_completion,
    marching_cubes_tactile_completion,
    qhull_tactile_completion
]
