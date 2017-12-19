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
import GPy as gpy


def _plot_triangles(points, tri):
    # import matplotlib.pyplot as plt

    # ax.triplot(points[:, 0], points[:, 1], tri.simplices.copy())

    # plt.figure()
    # plt.gca().set_aspect('equal')
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices, 'go-', lw=1.0)
    # plt.title('triplot of user-specified triangulation')
    # plt.xlabel('Longitude (degrees)')
    # plt.ylabel('Latitude (degrees)')
    #
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, cmap=plt.cm.Spectral)
    # # ax.plot_trisurf(ua, va, w, triangles=tri.triangles, cmap=plt.cm.Spectral)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    #
    # # The triangles in parameter space determine which x, y, z points are
    # # connected by an edge
    # # ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
    # ax.plot_trisurf(numpy.triangles=tri.simplices, cmap=plt.cm.Spectral)
    #
    # plt.show()

    pass


def _generate_ply_data(points, faces):
    vertices = [(point[0], point[1], point[2]) for point in points]
    faces = [(point,) for point in faces]

    vertices_np = numpy.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_np = numpy.array(faces, dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = plyfile.PlyElement.describe(vertices_np, 'vertex')
    face_element = plyfile.PlyElement.describe(faces_np, 'face')

    return plyfile.PlyData([vertex_element, face_element], text=True)


def _fast_triangulate(cloud):
    '''
    :param pcd_filename:
    '''
    # Convert point cloud to array
    points = cloud.to_array().astype(numpy.float32)

    # Triangulate points
    tri = scipy.spatial.Delaunay(points[:, 0:2])

    # Generate ply data
    return _generate_ply_data(points, tri.simplices)


def fast_triangulation(*pcd_filenames, **kwargs):
    suffix = kwargs.get("suffix", "")

    for pcd_filename in pcd_filenames:
        cloud = pcl.load(pcd_filename)

        plydata = _fast_triangulate(cloud)

        ply_filename = pcd_filename.replace(".pcd", suffix + ".ply")
        plydata.write(open(ply_filename, 'wb'))


def get_voxel_resolution(pc, patch_size, percent_patch_size):
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

    voxel_resolution = (1.0 * max_dim) / (percent_patch_size * patch_size)
    return voxel_resolution


def get_pointcloud_center(pc):
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


def create_voxel_grid_around_point_scaled(points, patch_center, voxel_resolution, num_voxels_per_dim, pc_center_in_voxel_grid):
    voxel_grid = numpy.zeros((num_voxels_per_dim, num_voxels_per_dim, num_voxels_per_dim, 1), dtype=numpy.float32)

    centered_scaled_points = numpy.floor(
        (points - numpy.array(patch_center) + numpy.array(
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
            numpy.zeros((csp_int.shape[0]), dtype=int))

    voxel_grid[mask] = 1

    return voxel_grid


def rescale_mesh(vertices,
                 patch_center,
                 voxel_resolution,
                 pc_center_in_voxel_grid):
    # Reverse of the following function solve for points
    return vertices * voxel_resolution - numpy.array(pc_center_in_voxel_grid) * voxel_resolution + numpy.array(patch_center)


def _complete_pointcloud_partial(pc, patch_size, percent_x, percent_y, percent_z, percent_patch_size):

    # TODO: Dynamic patch size based on point cloud size
    # patch_size = int(numpy.average(numpy.ptp(pc, axis=0)) * patch_size)

    vox_resolution = get_voxel_resolution(pc, patch_size, percent_patch_size)

    center = get_pointcloud_center(pc)

    pc_center_in_voxel_grid = (patch_size * percent_x, patch_size * percent_y, patch_size * percent_z)

    voxel_grid = create_voxel_grid_around_point_scaled(pc,
                                                       center,
                                                       vox_resolution,
                                                       patch_size,
                                                       pc_center_in_voxel_grid)

    vertices, faces = mcubes.marching_cubes(voxel_grid[:, :, :, 0], 0.5)
    vertices = rescale_mesh(vertices, center, vox_resolution, pc_center_in_voxel_grid)

    # Export to plyfile type
    return _generate_ply_data(vertices, faces)


def _smooth_ply(infilename, outfilename):
    # Initialize meshlabserver and meshlabxml script
    unsmoothed_mesh = meshlabxml.FilterScript(file_in=infilename, file_out=outfilename, ml_version="1.3.2")
    meshlabxml.smooth.laplacian(unsmoothed_mesh, iterations=6)
    unsmoothed_mesh.run_script()


def partial_completion(*pcd_filenames, **kwargs):
    patch_size = kwargs.get("patch_size", 120)
    percent_x = kwargs.get("percent_x", 0.5)
    percent_y = kwargs.get("percent_y", 0.5)
    percent_z = kwargs.get("percent_z", 0.45)
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    suffix = kwargs.get("suffix", "")

    for pcd_filename in pcd_filenames:

        cloud = pcl.load(pcd_filename)
        pc = cloud.to_array()

        plydata = _complete_pointcloud_partial(pc, patch_size, percent_x, percent_y, percent_z, percent_patch_size)

        ply_filename = pcd_filename.replace(".pcd", suffix + ".ply")
        plydata.write(open(ply_filename, 'wb'))

        _smooth_ply(ply_filename, ply_filename)
        # This does not work at the moment


def partial_completion_with_tactile(depth_pcd_filenames, tactile_pcd_filenames, **kwargs):
    patch_size = kwargs.get("patch_size", 120)
    percent_x = kwargs.get("percent_x", 0.5)
    percent_y = kwargs.get("percent_y", 0.5)
    percent_z = kwargs.get("percent_z", 0.45)
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    suffix = kwargs.get("suffix", "")

    for depth_pcd_filename, tactile_pcd_filename in zip(depth_pcd_filenames, tactile_pcd_filenames):
        depth_cloud = pcl.load(depth_pcd_filename)
        tactile_cloud = pcl.load(tactile_pcd_filename)

        cloud = pcl.load(pcd_filename)
        pc = cloud.to_array()

        plydata = _complete_pointcloud_partial(pc, patch_size, percent_x, percent_y, percent_z, percent_patch_size)

        ply_filename = pcd_filename.replace(".pcd", suffix + ".ply")
        plydata.write(open(ply_filename, 'wb'))

        _smooth_ply(ply_filename, ply_filename)
        # This does not work at the moment


def _qhull_pointcloud_completion(cloud):
    points = cloud.to_array()

    hull = ConvexHull(points)

    # Fix inverted normals from pyhull
    hull.vertices = [vertex[::-1] for vertex in hull.vertices]

    return _generate_ply_data(points, hull.vertices)


def qhull_completion(*pcd_filenames, **kwargs):
    suffix = kwargs.get("suffix", "")

    for pcd_filename in pcd_filenames:
        cloud = pcl.load(pcd_filename)

        plydata = _qhull_pointcloud_completion(cloud)

        ply_filename = pcd_filename.replace(".pcd", suffix + ".ply")
        plydata.write(open(ply_filename, 'wb'))


def _gaussian_process_pointcloud_completion(cloud, patch_size, percent_x, percent_y, percent_z, percent_patch_size):
    # Evaluate https://github.com/kshitijl/pycuda-gaussian-processes for potential speedup
    points = cloud.to_array()

    # Generate gaussian process
    kernel_hyps = [1., 3.]
    meas_variance = 0.1
    kernel = gpy.kern.RBF(input_dim=3, variance=kernel_hyps[0], lengthscale=kernel_hyps[1])
    points_positive = numpy.copy(points)
    points_positive[:, 2] += 0.01
    points_negative = numpy.copy(points)
    points_negative[:, 2] -= 0.01
    points = numpy.concatenate([points_positive, points_negative], axis=0)

    sdf_meas = numpy.ones((points.shape[0], 1))
    for i in range(points.shape[0]/2):
        sdf_meas[i] *= -1

    gp = gpy.models.GPRegression(points, sdf_meas, kernel)
    gp.optimize()

    # Create voxel grid to use for gaussian process
    center_point = get_pointcloud_center(points)
    voxel_resolution = get_voxel_resolution(points, patch_size, percent_patch_size)

    # Where am I putting my point cloud relative to the center of my voxel grid
    # ex. (20, 20, 20) or (20, 20, 18)
    pc_center_in_voxel_grid = (patch_size * percent_x, patch_size * percent_y, patch_size * percent_z)

    voxel_grid = create_voxel_grid_around_point_scaled(points,
                                                       center_point,
                                                       voxel_resolution,
                                                       patch_size,
                                                       pc_center_in_voxel_grid)

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
    prediction = gp.predict(points, full_cov=True)

    output_grid = numpy.zeros((20, 20, 20))
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


def gaussian_process_general_completion(*pcd_filenames, **kwargs):
    patch_size = kwargs.get("patch_size", 120)
    percent_x = kwargs.get("percent_x", 0.5)
    percent_y = kwargs.get("percent_y", 0.5)
    percent_z = kwargs.get("percent_z", 0.45)
    percent_patch_size = kwargs.get("percent_patch_size", 0.8)
    suffix = kwargs.get("suffix", "")

    for pcd_filename in pcd_filenames:
        cloud = pcl.load(pcd_filename)

        plydata = _gaussian_process_pointcloud_completion(cloud, patch_size, percent_x, percent_y, percent_z, percent_patch_size)

        ply_filename = pcd_filename.replace(".pcd", suffix + ".ply")
        plydata.write(open(ply_filename, 'wb'))


def plot_sdf_meas(points, sdf_meas):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    c = sdf_meas[:, 0]

    ax.scatter(x, y, z, c=c, cmap=plt.hot())
    plt.show()


def _gaussian_process_pointcloud_depth_tactile_completion(depth_cloud, tactile_cloud, patch_size, percent_x, percent_y, percent_z, percent_patch_size):
    # Evaluate https://github.com/kshitijl/pycuda-gaussian-processes for potential speedup
    depth_points = depth_cloud.to_array()
    tactile_points = tactile_cloud.to_array()

    # Generate gaussian process
    kernel_hyps = [0.1, 0.3]
    meas_variance = 0.1
    kernel = gpy.kern.RBF(input_dim=3, variance=kernel_hyps[0], lengthscale=kernel_hyps[1])

    # Copy depth points
    points_positive = numpy.copy(depth_points)
    points_positive[:, 2] += 0.01
    points_negative = numpy.copy(depth_points)
    # points_negative[:, 2] -= 0.01
    depth_points = numpy.concatenate([points_negative, points_positive], axis=0)
    sdf_meas_depth = numpy.ones((depth_points.shape[0], 1))
    for i in range(depth_points.shape[0]/2):
        sdf_meas_depth[i] *= 0

    out = _complete_pointcloud_partial(depth_points, patch_size, percent_x, percent_y, percent_z, percent_patch_size)
    out.write(open("depth.ply", 'w'))

    # Copy tactile points
    points_positive = numpy.copy(tactile_points)
    points_positive[:, 2] += 0.01
    points_negative = numpy.copy(tactile_points)
    # points_negative[:, 2] -= 0.01
    tactile_points = numpy.concatenate([points_positive, points_negative], axis=0)
    sdf_meas_tactile = numpy.ones((tactile_points.shape[0], 1))
    for i in range(tactile_points.shape[0]/2):
        sdf_meas_tactile[i] *= 0

    out = _complete_pointcloud_partial(tactile_points, patch_size, percent_x, percent_y, percent_z, percent_patch_size)
    out.write(open("tactile.ply", 'w'))

    points = numpy.concatenate([depth_points, tactile_points], axis=0)
    sdf_meas = numpy.concatenate([sdf_meas_depth, sdf_meas_tactile], axis=0)

    # plot_sdf_meas(points, sdf_meas)

    gp = gpy.models.GPRegression(points, sdf_meas, kernel)
    gp.optimize()

    # Create voxel grid to use for gaussian process
    center_point = get_pointcloud_center(points)
    voxel_resolution = get_voxel_resolution(points, patch_size, percent_patch_size)

    # Where am I putting my point cloud relative to the center of my voxel grid
    # ex. (20, 20, 20) or (20, 20, 18)
    pc_center_in_voxel_grid = (patch_size * percent_x, patch_size * percent_y, patch_size * percent_z)

    voxel_grid = create_voxel_grid_around_point_scaled(points,
                                                       center_point,
                                                       voxel_resolution,
                                                       patch_size,
                                                       pc_center_in_voxel_grid)

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
    prediction = gp.predict(points, full_cov=True)

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


if __name__ == "__main__":
    fast_triangulation("/home/david/Downloads/bun0.pcd", suffix="_triangulation")
    qhull_completion("/home/david/Downloads/bun0.pcd", suffix="_qhull")
    partial_completion("/home/david/Downloads/bun0.pcd", suffix="_partial", patch_size=40)
    # gaussian_process_general_completion("/home/david/Downloads/pringles_original_2_y17_m10_d30_h13_m53_s20/out.pcd", suffix="_gaussian", patch_size=20)
    gaussian_process_depth_tactile_completion(
        ["/home/david/Downloads/pringles_original_2_y17_m10_d30_h13_m53_s20/out.pcd"],
        ["/home/david/Downloads/pringles_original_2_y17_m10_d30_h13_m53_s20/tactile_cf.pcd"],
        suffix="_gaussian",
        patch_size=20,
        percent_z=0.5
    )
