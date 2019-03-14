import shape_msgs.msg
import numpy
import plyfile
import geometry_msgs.msg
import std_msgs.msg
import copy
import collada
import sensor_msgs.point_cloud2
import numpy as np


def mesh_msg_to_dae(mesh_msg):
    """
    Converts a ROS mesh into a collada object which can be then written to a file
    :type mesh_msg: shape_msgs.msg.Mesh
    :rtype collada.Collada
    """
    vs = []
    for vert in mesh_msg.vertices:
        vs.append((vert.x, vert.y, vert.z))
    vertices = numpy.array(vs)

    ts = []
    for tri in mesh_msg.triangles:
        ts.append(tri.vertex_indices)
    triangles = numpy.array(ts)

    mesh = collada.Collada()

    vert_src = collada.source.FloatSource("verts-array", vertices, ('X','Y','Z'))
    geom = collada.geometry.Geometry(mesh, "geometry0", "curvox_mesh", [vert_src])

    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', "#verts-array")

    triset = geom.createTriangleSet(numpy.copy(triangles), input_list, "")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    geomnode = collada.scene.GeometryNode(geom, [])
    node = collada.scene.Node("curvox_mesh", children=[geomnode])

    myscene = collada.scene.Scene("mcubes_scene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene

    return mesh


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


def ply_to_pointcloud_msg(ply, downsample=0, frame_id='/world'):
    """
    :type ply: plyfile.PlyData
    :rtype sensor_msgs.msg.PointCloud2
    """

    points = [(vertex['x'], vertex['y'], vertex['z']) for vertex in ply.elements[0].data]

    header = std_msgs.msg.Header()
    header.frame_id = frame_id
    pcl_arr = sensor_msgs.point_cloud2.create_cloud_xyz32(header, np.array(points))

    return pcl_arr
