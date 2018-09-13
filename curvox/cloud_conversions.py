import pcl
import numpy as np


def write_pcd_with_normals(points, normals, output_pcd_filepath):

    with open(output_pcd_filepath, 'w') as new_pcd_file:
        new_pcd_file.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z normal_x normal_y normal_z\n"
            "SIZE 4 4 4 4 4 4\n"
            "TYPE F F F F F F\n"
            "COUNT 1 1 1 1 1 1\n"
            "WIDTH {0}\n"
            "HEIGHT 1\n"
            "POINTS {0}\n"
            "DATA ascii\n"
                .format(points.shape[0])
        )

        for point, normal in zip(points, normals):
            new_pcd_file.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], normal[0], normal[1], normal[2]))


def compute_tactile_normals(cloud):
    points = cloud.to_array()
    num_points = points.shape[0]
    # n x 1 magnitude of each point
    magnitudes = np.linalg.norm(points, axis=1).reshape((num_points, 1))
    # divide points by per point magnitude, and flip sign to point back at origin
    normals = points / magnitudes

    return normals


def compute_depth_normals(pcd, ksearch, search_radius):
    points = pcd.to_array()
    # Convert PCD to PCD with normals
    normals = pcd.calc_normals(ksearch=ksearch, search_radius=search_radius)

    return normals


def calculate_normals_from_depth_and_tactile(depth_cloud, tactile_cloud, downsampled_pointcloud_size):
    v_points = depth_cloud.to_array()
    v_normals = compute_depth_normals(depth_cloud, ksearch=10, search_radius=0)

    depth_downsample_factor = v_points.shape[0] / downsampled_pointcloud_size

    v_points = v_points[::depth_downsample_factor]
    v_normals = v_normals[::depth_downsample_factor]

    t_points = tactile_cloud.to_array()
    t_normals = compute_tactile_normals(tactile_cloud)

    vt_points = np.concatenate([v_points, t_points])
    vt_normals = np.concatenate([v_normals, t_normals])

    return vt_points, vt_normals


def pcd_to_np(pcd_filename):
    """
    Read in PCD then return nx3 numpy array

    :type pcd_filename: str
    :rtype numpy.ndarray
    """

    pcd = pcl.load(pcd_filename)
    return pcl_to_np(pcd)


def pcl_to_np(pointcloud):
    """
    Convert PCL pointcloud to numpy nx3 numpy array

    :type pointcloud: pcl.PointCloud
    :rtype numpy.ndarray
    """

    pc_nx3 = pointcloud.to_array()
    return pc_nx3


def np_to_pcl(pc_np):
    """
    Convert nx3 numpy array to PCL pointcloud

    :type pc_np: numpy.ndarray
    :rtype pcl.PointCloud
    """

    print(pc_np.shape)
    new_pcd = pcl.PointCloud(np.array(pc_np, np.float32))
    return new_pcd


