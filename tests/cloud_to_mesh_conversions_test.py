from curvox.cloud_to_mesh_conversions import \
    delaunay_completion, marching_cubes_completion, qhull_completion, gaussian_process_completion \
    delaunay_tactile_completion, marching_cubes_tactile_completion, qhull_tactile_completion, gaussian_process_tactile_completion
import os
import pcl
import time


def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    fn_name = fn.__module__ + "." + fn.__name__
    print fn_name + ": " + str(end-start) + "s"
    return results


def test_depth_completions():
    bunny_cloud = pcl.load('resources/bunny.pcd')
    bunny_points = bunny_cloud.to_array()

    # Testing depth completions only
    delaunay_ply = time_fn(delaunay_completion, bunny_points)
    smooth_delaunay_ply = time_fn(delaunay_completion, bunny_points, smooth=True)

    delaunay_ply = time_fn(marching_cubes_completion, bunny_points)
    smooth_delaunay_ply = time_fn(marching_cubes_completion, bunny_points, smooth=True)

    delaunay_ply = time_fn(qhull_completion, bunny_points)
    smooth_delaunay_ply = time_fn(qhull_completion, bunny_points, smooth=True)

    delaunay_ply = time_fn(gaussian_process_completion, bunny_points)
    smooth_delaunay_ply = time_fn(gaussian_process_completion, bunny_points, smooth=True)


def test_tactile_completions():
    depth_cloud = pcl.load('resources/depth_cloud_cf.pcd')
    tactile_cloud = pcl.load('resources/tactile_cf.pcd')
    depth_points = depth_cloud.to_array()
    tactile_points = tactile_cloud.to_array()

    # Testing depth completions only
    delaunay_ply = time_fn(delaunay_tactile_completion, depth_points, tactile_points)
    smooth_delaunay_ply = time_fn(delaunay_tactile_completion, depth_points, tactile_points, smooth=True)

    delaunay_ply = time_fn(marching_cubes_tactile_completion, depth_points, tactile_points)
    smooth_delaunay_ply = time_fn(marching_tactile_cubes_completion, depth_points, tactile_points, smooth=True)

    delaunay_ply = time_fn(qhull_tactile_completion, depth_points, tactile_points)
    smooth_delaunay_ply = time_fn(qhull_tactile_completion, depth_points, tactile_points, smooth=True)

    delaunay_ply = time_fn(gaussian_process_tactile_completion, depth_points, tactile_points)
    smooth_delaunay_ply = time_fn(gaussian_process_tactile_completion, depth_points, tactile_points, smooth=True)


def test_load_and_save():
    pass