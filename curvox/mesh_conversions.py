import shape_msgs.msg
import numpy
import plyfile
import geometry_msgs.msg
import copy
import pcl


def mesh_msg_to_ply(mesh_msg):
    """
    :type mesh_msg: shape_msgs.msg.Mesh
    :rtype plyfile.PlyData
    """

    # vertex = numpy.array([(0, 0, 0),
    #                       (0, 1, 1),
    #                       (1, 0, 1),
    #                       (1, 1, 0)],
    #                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # face = numpy.array([([0, 1, 2],),
    #                     ([0, 2, 3],),
    #                     ([0, 1, 3],),
    #                     ([1, 2, 3],)],
    #                     dtype=[('vertex_indices', 'i4', (3,))])

    vertices = [(x.x, x.y, x.z) for x in mesh_msg.vertices]
    faces = [(x.vertex_indices,) for x in mesh_msg.triangles]

    vertices_np = numpy.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_np = numpy.array(faces, dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = plyfile.PlyElement.describe(vertices_np, 'vertex')
    face_element = plyfile.PlyElement.describe(faces_np, 'face')

    return plyfile.PlyData([vertex_element, face_element], text=True)


def write_mesh_msg_to_ply_filepath(mesh_msg, output_filepath):
    """
    :type mesh_msg: shape_msgs.msg.Mesh
    :type output_filepath: str
    """
    mesh_msg_to_ply(mesh_msg).write(output_filepath)


def ply_to_mesh_msg(ply):
    """
    :type ply: plyfile.PlyData
    :rtype shape_msgs.msg.Mesh
    """

    mesh_msg = shape_msgs.msg.Mesh()

    vertices = ply.elements[0]
    mesh_msg.vertices = [geometry_msgs.msg.Point(vertex['x'], vertex['y'], vertex['z']) for vertex in vertices.data]

    triangles = ply.elements[1]
    mesh_msg.triangles = [shape_msgs.msg.MeshTriangle(*triangle) for triangle in triangles.data]

    return mesh_msg


def read_mesh_msg_from_ply_filepath(ply_filepath):
    """
    :type ply_filepath: str
    :rtype shape_msgs.msg.Mesh
    """

    ply = plyfile.PlyData.read(ply_filepath)
    return ply_to_mesh_msg(ply)


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


def merge_pcd_files(out_filename, *pcd_filenames):

    pcd_objs = map(pcl.load, pcd_filenames)
    pcd_arrs = map(lambda x: x.to_array(), pcd_objs)
    total_pcd_arr = numpy.concatenate(pcd_arrs, axis=0)
    pcl.save(pcl.PointCloud(total_pcd_arr), out_filename)
