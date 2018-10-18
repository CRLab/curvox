import sensor_msgs.msg
import sensor_msgs.point_cloud2
import tf
import tf_conversions
import tf2_ros
import std_msgs.msg
import rospy
import numpy as np
import pcl
import numba
import ros_numpy
import operator


@numba.jit
def cloud_msg_to_np(msg):
    """
    Take a ros pointclud message and convert it to
    an nx3 numpy ndarray.

    :type msg: sensor_msg.msg.PointCloud2
    :rtype numpy.ndarray
    """
    pc = ros_numpy.numpify(msg)
    num_pts = reduce(operator.mul, pc.shape, 1)
    points = np.zeros((num_pts, 3))
    points[:, 0] = pc['x'].reshape(num_pts)
    points[:, 1] = pc['y'].reshape(num_pts)
    points[:, 2] = pc['z'].reshape(num_pts)

    return np.array(points, dtype=np.float32)


def np_to_cloud_msg(pc_np, frame_id):
    """
    :type pc_np: numpy.ndarray
    :param pc_np: A nx3 pointcloud
    :rtype sensor_msg.msg.PointCloud2
    """

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    cloud_msg = sensor_msgs.point_cloud2.create_cloud_xyz32(header, pc_np)

    return cloud_msg


@numba.jit
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


def capture_cloud(pc_topic):
    """
    :type pc_topic: str
    :param pc_topic: Point cloud topic
    :return: PointCloud of the captured ros topic
    :rtype: pcl.PointCloud
    """
    if not rospy.core.is_initialized():
        rospy.init_node("capture_pointcloud", anonymous=True)

    pointcloud_msg = rospy.wait_for_message(pc_topic, sensor_msgs.msg.PointCloud2)
    np_pc = cloud_msg_to_np(pointcloud_msg)

    pcl_pc = pcl.PointCloud()
    pcl_pc.from_array(np_pc)

    return pcl_pc


def capture_cloud_and_transform(pc_topic, source_frame, target_frame):
    """
    :param pc_topic:
    :type pc_topic: str
    :param source_frame:
    :type source_frame: str
    :param target_frame:
    :type target_frame: str
    :return:
    :rtype: pcl.PointCloud
    """
    pcl_pc = capture_cloud(pc_topic)

    # Transform cloud into correct frame
    tf_msg = capture_tf_msg(source_frame, target_frame)

    cf2obj_tf_msg = tf_conversions.posemath.fromTf(tf_msg)
    cf2obj_tf_mat = tf_conversions.posemath.toMatrix(cf2obj_tf_msg)

    np_pc = pcl_pc.to_array()
    np_pc = transform_cloud(np_pc, cf2obj_tf_mat)
    np_pc = np_pc.astype(np.float32)

    pcl_pc = pcl.PointCloud()
    pcl_pc.from_array(np_pc)

    return pcl_pc


def capture_tf_msg(source_frame, target_frame):
    listener = tf.TransformListener()

    # Transform cloud into correct frame
    try:
        now = rospy.Time(0)
        listener.waitForTransform(target_frame, source_frame, now, rospy.Duration(5))
        tf_msg = listener.lookupTransform(target_frame, source_frame, now)
    except tf2_ros.TransformException as e:
        rospy.logerr("Exception: {}, returning raw pointcloud".format(e))
        raise e

    return tf_msg


def capture_tf_msg_np(source_frame, target_frame):
    tf_msg = capture_tf_msg(source_frame, target_frame)
    src2tgt_tf_msg = tf_conversions.posemath.fromTf(tf_msg)
    src2tgt_tf_mat = tf_conversions.posemath.toMatrix(src2tgt_tf_msg)

    return src2tgt_tf_mat
