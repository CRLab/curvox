import sensor_msgs.point_cloud2 as pcl2
import std_msgs
import rospy
import numpy as np


def cloud_msg_to_np(msg):
    """
    Take a ros pointclud message and convert it to
    an nx3 numpy ndarray.

    :type msg: sensor_msg.msg.PointCloud2
    :rtype numpy.ndarray
    """

    num_pts = msg.width*msg.height
    out = np.zeros((num_pts, 4))
    count = 0
    for point in pcl2.read_points(msg, skip_nans=False):
        out[count] = point
        count += 1

    # if there were nans, we need to resize cloud to skip them.
    out = out[:count, 0:3]
    return out


def np_to_cloud_msg(pc_np, frame_id):
    """
    :type pc_np: numpy.ndarray
    :param pc_np: A nx3 pointcloud
    :rtype sensor_msg.msg.PointCloud2
    """

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    cloud_msg = pcl2.create_cloud_xyz32(header, pc_np)

    return cloud_msg


def transform_cloud(pc, transform):
    """
    :type pc: numpy.ndarray
    :type transform: numpy.ndarray
    :param transform: A 4x4 homogenous transform to apply to the cloud
    :param pc: A nx3 pointcloud
    :rtype numpy.ndarray
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
