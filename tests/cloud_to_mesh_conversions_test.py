from curvox.cloud_to_mesh_conversions import \
    delaunay_completion, marching_cubes_completion, qhull_completion, gaussian_process_completion, \
    delaunay_tactile_completion, marching_cubes_tactile_completion, qhull_tactile_completion, gaussian_process_tactile_completion
from curvox.utils import time_fn
import pcl


def _depth_completion_test(completion_method, completion_name, smoothed):
    bunny_cloud = pcl.load('resources/bunny.pcd')
    bunny_points = bunny_cloud.to_array()

    # Testing depth completions only
    ply_data = time_fn(completion_method, bunny_points)

    smoothed_str = ["unsmoothed", "smoothed"][smoothed]
    ply_data.write(open("resources/bunny_{}_{}.ply".format(completion_name, smoothed_str), 'w'))


def test_delaunay_completion():
    _depth_completion_test(delaunay_completion, "delaunay", smoothed=False)


def test_delaunay_smoothed_completion():
    _depth_completion_test(delaunay_completion, "delaunay", smoothed=True)


def test_marching_cubes_completion():
    _depth_completion_test(marching_cubes_completion, "marching_cubes", smoothed=False)


def test_marching_cubes_completion_smoothed():
    _depth_completion_test(marching_cubes_completion, "marching_cubes", smoothed=True)


def test_qhull_completion():
    _depth_completion_test(qhull_completion, "qhull", smoothed=False)


def test_qhull_completion_smoothed():
    _depth_completion_test(qhull_completion, "qhull", smoothed=True)


def test_gaussian_process_completion():
    _depth_completion_test(gaussian_process_completion, "gaussian_process", smoothed=False)


def test_gaussian_process_completion_smoothed():
    _depth_completion_test(gaussian_process_completion, "gaussian_process", smoothed=True)


def test_tactile_completions():
    depth_cloud = pcl.load('resources/depth_cloud_cf.pcd')
    tactile_cloud = pcl.load('resources/tactile_cf.pcd')
    depth_points = depth_cloud.to_array()
    tactile_points = tactile_cloud.to_array()

    # Testing depth completions only
    delaunay_ply = time_fn(delaunay_tactile_completion, depth_points, tactile_points)
    # smooth_delaunay_ply = time_fn(delaunay_tactile_completion, depth_points, tactile_points, smooth=True)

    delaunay_ply = time_fn(marching_cubes_tactile_completion, depth_points, tactile_points)
    # smooth_delaunay_ply = time_fn(marching_cubes_tactile_completion, depth_points, tactile_points, smooth=True)

    delaunay_ply = time_fn(qhull_tactile_completion, depth_points, tactile_points)
    # smooth_delaunay_ply = time_fn(qhull_tactile_completion, depth_points, tactile_points, smooth=True)

    delaunay_ply = time_fn(gaussian_process_tactile_completion, depth_points, tactile_points)
    # smooth_delaunay_ply = time_fn(gaussian_process_tactile_completion, depth_points, tactile_points, smooth=True)


def test_load_and_save():
    pass