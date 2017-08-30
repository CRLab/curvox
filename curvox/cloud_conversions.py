import pcl
import numpy as np


def pcd_to_np(pcd_filename):
    """
    :type pcd_filename: str
    """
    # Read in PCD then return nx3 numpy array
    pcd = pcl.load(pcd_filename)
    return pcl_to_np(pcd)


def pcl_to_np(pointcloud):
    """
    :type pointcloud: pcl.PointCloud
    """
    # Convert PCL pointcloud to numpy nx3 numpy array
    pc_nx3 = pointcloud.to_array()
    return pc_nx3


def np_to_pcl(pc_np):
    """
    :type pc_np: numpy.ndarray
    """
    # Convert nx3 numpy array to PCL pointcloud
    new_pcd = pcl.PointCloud()
    new_pcd.from_array(pc_np)
    return new_pcd


def transform_cloud(pc, transform):
    """
    :type pc: np.ndarray
    :type transform: numpy.ndarray
    :param transform: A 4x4 homogenous transform to apply to the cloud
    :param pc: A nx3 pointcloud
    """

    if len(pc.shape) != 2:
        print("Warning, pc shape length should be 2")
    if pc.shape[1] != 3:
        print("Warning: pc.shape[1] != 3 your pointcloud may be transposed!!!")

    num_pts = pc.shape[0]
    homogenous_coords = np.ones((num_pts, 4))
    homogenous_coords[:, 0:3] = pc

    #need points to be 4xn
    homogenous_coords = homogenous_coords.T

    out_4xn = np.dot(transform, homogenous_coords)

    out_nx3 = out_4xn.T[:, 0:3]

    return out_nx3

