import shape_msgs.msg
import numpy
from plyfile import PlyData, PlyElement
import plyfile
import geometry_msgs.msg

def write_mesh_msg_to_ply_filepath(mesh_msg, output_filepath):
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

    vertex_element = PlyElement.describe(vertices_np, 'vertex')
    face_element = PlyElement.describe(faces_np, 'face')

    PlyData([vertex_element, face_element], text=True).write(output_filepath)


def read_mesh_msg_from_ply_filepath(ply_filepath):
    p = plyfile.PlyData.read(ply_filepath)
    vertices = p.elements[0]

    # Build up a ROS shape_msgs.msg.Mesh so that it can be returned to the client
    mesh_msg = shape_msgs.msg.Mesh()
    for i in range(vertices.data.shape[0]):
        point = geometry_msgs.msg.Point(vertices.data[i]['x'], vertices.data[i]['y'], vertices.data[i]['z'])
        mesh_msg.vertices.append(point)
        
    triangles = p.elements[1]
    for i in range(triangles.data.shape[0]):
        t = shape_msgs.msg.MeshTriangle(*triangles.data[i])
        mesh_msg.triangles.append(t)

    return mesh_msg