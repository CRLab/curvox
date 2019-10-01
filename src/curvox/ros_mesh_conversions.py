import shape_msgs.msg
import numpy
import plyfile
import geometry_msgs.msg
import std_msgs.msg
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


def ply_to_pointcloud_msg(ply, frame_id='/world'):
    """
    :type ply: plyfile.PlyData
    :rtype sensor_msgs.msg.PointCloud2
    """

    points = [(vertex['x'], vertex['y'], vertex['z']) for vertex in ply.elements[0].data]

    header = std_msgs.msg.Header()
    header.frame_id = frame_id
    pcl_arr = sensor_msgs.point_cloud2.create_cloud_xyz32(header, np.array(points))

    return pcl_arr

