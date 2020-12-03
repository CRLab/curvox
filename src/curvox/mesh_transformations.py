import numpy
import plyfile
import copy
import numpy as np
import pypcd


def transform_ply(ply, transform):
    """
    :type ply: plyfile.PlyData
    :type transform: numpy.ndarray
    :param transform: A 4x4 homogenous transform to apply to the ply mesh
    :rtype plyfile.PlyData
    """

    #Translate ply into 4xN array
    mesh_vertices = numpy.ones((4, ply['vertex']['x'].shape[0]))
    mesh_vertices[0, :] = ply['vertex']['x']
    mesh_vertices[1, :] = ply['vertex']['y']
    mesh_vertices[2, :] = ply['vertex']['z']

    #Create new 4xN transformed array
    transformed_mesh = numpy.dot(transform, mesh_vertices)

    transformed_ply = copy.deepcopy(ply)

    #Write transformed vertices back to ply
    transformed_ply['vertex']['x'] = transformed_mesh[0, :]
    transformed_ply['vertex']['y'] = transformed_mesh[1, :]
    transformed_ply['vertex']['z'] = transformed_mesh[2, :]

    return transformed_ply


def ply_to_pcl(ply):
    """
    :type ply: plyfile.PlyData
    :rtype sensor_msgs.msg.PointCloud2
    """

    points = np.array([(vertex['x'], vertex['y'], vertex['z']) for vertex in ply.elements[0].data])

    pc = pypcd.Pointcloud.from_array(points)

    return pc


def ply_to_np(ply):
    """
    :type ply: plyfile.PlyData
    :rtype sensor_msgs.msg.PointCloud2
    """

    points = np.array([(vertex['x'], vertex['y'], vertex['z']) for vertex in ply.elements[0].data])

    return points
