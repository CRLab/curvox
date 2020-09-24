import functools
import numpy as np
import operator
import struct

import numba
import pcl
import ctypes

try:
    import ros_numpy
    import rospy
    import sensor_msgs.msg
    import sensor_msgs.point_cloud2
    import geometry_msgs.msg
    import shape_msgs.msg
    import std_msgs.msg
    import tf
    import tf2_ros
    import tf_conversions

    _DATATYPES = {}
    _DATATYPES[sensor_msgs.msg.PointField.INT8]    = ('b', 1)
    _DATATYPES[sensor_msgs.msg.PointField.UINT8]   = ('B', 1)
    _DATATYPES[sensor_msgs.msg.PointField.INT16]   = ('h', 2)
    _DATATYPES[sensor_msgs.msg.PointField.UINT16]  = ('H', 2)
    _DATATYPES[sensor_msgs.msg.PointField.INT32]   = ('i', 4)
    _DATATYPES[sensor_msgs.msg.PointField.UINT32]  = ('I', 4)
    _DATATYPES[sensor_msgs.msg.PointField.FLOAT32] = ('f', 4)
    _DATATYPES[sensor_msgs.msg.PointField.FLOAT64] = ('d', 8)
except:
    pass


@numba.jit
def cloud_msg_to_np(msg):
    """
    Take a ros pointclud message and convert it to
    an nx3 np ndarray.

    :type msg: sensor_msg.msg.PointCloud2
    :rtype np.ndarray
    """
    pc = ros_numpy.numpify(msg)
    num_pts = functools.reduce(operator.mul, pc.shape, 1)
    points = np.zeros((num_pts, 3))
    points[:, 0] = pc['x'].reshape(num_pts)
    points[:, 1] = pc['y'].reshape(num_pts)
    points[:, 2] = pc['z'].reshape(num_pts)

    return np.array(points, dtype=np.float32)


def np_to_cloud_msg(pc_np, frame_id):
    """
    :param frame_id:
    :type frame_id: str
    :type pc_np: np.ndarray
    :param pc_np: A nx3 pointcloud
    :rtype sensor_msg.msg.PointCloud2
    """

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    # data = np.zeros(pc_np.shape[0], dtype=[
    #     ('x', np.float32),
    #     ('y', np.float32),
    #     ('vectors', np.float32, (3,))
    # ])
    # data['x'] = np.arange(100)
    # data['y'] = data['x'] * 2
    # data['vectors'] = np.arange(100)[:, np.newaxis]
    #
    # msg = ros_numpy.msgify(PointCloud2, data)
    cloud_msg = sensor_msgs.point_cloud2.create_cloud_xyz32(header, pc_np)

    return cloud_msg


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    """
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    """
    assert (points.shape[0] == colors.shape[0])

    header = std_msgs.msg.Header()

    if stamp:
        header.stamp = stamp
    if frame_id:
        header.frame_id = frame_id
    if seq:
        header.seq = seq

    new_colors = []
    for r, g, b in colors:
        new_colors.append(struct.unpack('I', struct.pack('BBBB', r, g, b, 0))[0])

    new_colors = np.array(new_colors, dtype=np.uint32)
    points = points.astype(np.float32)
    x, y, z = points.T

    xyzrgb = np.rec.fromarrays([x, y, z, new_colors], names='x,y,z,colors')

    fields = [
        sensor_msgs.msg.PointField('x', 0, sensor_msgs.msg.PointField.FLOAT32, 1),
        sensor_msgs.msg.PointField('y', 4, sensor_msgs.msg.PointField.FLOAT32, 1),
        sensor_msgs.msg.PointField('z', 8, sensor_msgs.msg.PointField.FLOAT32, 1),
        sensor_msgs.msg.PointField('rgb', 12, sensor_msgs.msg.PointField.UINT32, 1),
    ]

    msg = sensor_msgs.point_cloud2.create_cloud(header, fields, xyzrgb)

    return msg


def cloud_msg_to_json_dict(cloud_msg):
    """

    :param cloud_msg:
    :type cloud_msg: sensor_msgs.msg.PointCloud2
    :return:
    :rtype: dict
    """
    data = cloud_msg.data.encode('base64')
    msg = dict(
        header=dict(
            seq=cloud_msg.header.seq,
            frame_id=cloud_msg.header.frame_id
        ),
        height=cloud_msg.height,
        width=cloud_msg.width,
        fields=[
            dict(
                name=field.name,
                offset=field.offset,
                datatype=field.datatype,
                count=field.count
            ) for field in cloud_msg.fields
        ],
        is_bigendian=cloud_msg.is_bigendian,
        point_step=cloud_msg.point_step,
        row_step=cloud_msg.row_step,
        data=data,
        is_dense=cloud_msg.is_dense
    )
    return msg


def json_dict_to_mesh_msg(mesh_json_dict):
    """
    In [5]: msg_payload[0]["mesh"].keys()
    Out[5]: [u'vertices', u'triangles']

    In [6]: msg_payload[0]["mesh"]["vertices"][0]
    Out[6]:
    {u'x': -0.7574082016944885,
     u'y': -0.7571043968200684,
     u'z': 1.7426010370254517}

    In [7]: msg_payload[0]["mesh"]["triangles"][0]
    Out[7]: {u'vertex_indices': [2, 1, 0]}


    :param cloud_json_dict:
    :return:
    """
    mesh_msg = shape_msgs.msg.Mesh()
    for vertex in mesh_json_dict["vertices"]:
        mesh_msg.vertices.append(geometry_msgs.msg.Point(x=vertex["x"], y=vertex["y"], z=vertex["z"]))

    for triangle in mesh_json_dict["triangles"]:
        mesh_msg.triangles.append(shape_msgs.msg.MeshTriangle(vertex_indices=triangle["vertex_indices"]))

    return mesh_msg


@numba.jit
def transform_cloud(pc, transform):
    """
    :type pc: np.ndarray
    :type transform: np.ndarray
    :param transform: A 4x4 homogenous transform to apply to the cloud
    :param pc: A nx3 pointcloud
    :rtype np.ndarray
    """

    if len(pc.shape) != 2:
        print("Warning, pc shape length should be 2")
    if pc.shape[1] != 3:
        print("Warning: pc.shape[1] != 3 your pointcloud may be transposed!!!")

    num_pts = pc.shape[0]
    homogenous_coords = np.ones((num_pts, 4))
    homogenous_coords[:, 0:3] = pc

    # need points to be 4xn
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


def transform_cloud_by_frame(pcl_pc, source_frame, target_frame):
    '''
    Transform the cloud between two frames
    '''

    # Transform cloud into correct frame
    tf_msg = capture_tf_msg(source_frame, target_frame)

    cf2obj_tf_msg = tf_conversions.posemath.fromTf(tf_msg)
    cf2obj_tf_mat = tf_conversions.posemath.toMatrix(cf2obj_tf_msg)

    try:
        np_pc = pcl_pc.to_array()
    except:
        np_pc = pcl_pc
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
