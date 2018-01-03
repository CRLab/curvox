import math
import meshlabxml
import os
import tempfile
import plyfile
import numpy as np
import numba
import binvox_rw
import subprocess


def print_hausdorff(hausdorff_distance):
    for key, value in hausdorff_distance.items():
        print('{}: {}'.format(key, value))


@numba.njit
def minmax(array):
    # Ravel the array and return early if it's empty
    array = array.ravel()
    length = array.size
    if not length:
        return

    # We want to process two elements at once so we need
    # an even sized array, but we preprocess the first and
    # start with the second element, so we want it "odd"
    odd = length % 2
    if not odd:
        length -= 1

    # Initialize min and max with the first item
    minimum = maximum = array[0]

    i = 1
    while i < length:
        # Get the next two items and swap them if necessary
        x = array[i]
        y = array[i+1]
        if x > y:
            x, y = y, x
        # Compare the min with the smaller one and the max
        # with the bigger one
        minimum = min(x, minimum)
        maximum = max(y, maximum)
        i += 2

    # If we had an even sized array we need to compare the
    # one remaining item too.
    if not odd:
        x = array[length]
        minimum = min(x, minimum)
        maximum = max(x, maximum)

    return minimum, maximum


def hausdorff_distance_one_direction(mesh1_filepath, mesh2_filepath):
    script = meshlabxml.create.FilterScript(file_in=[mesh1_filepath, mesh2_filepath], ml_version='1.3.2')
    meshlabxml.sampling.hausdorff_distance(script)
    script.run_script(print_meshlabserver_output=False)
    return script.hausdorff_distance


@numba.jit
def hausdorff_distance_bi(mesh1_filepath, mesh2_filepath):
    # get hausdorff dist from meshlab server
    hd_ab = hausdorff_distance_one_direction(mesh1_filepath, mesh2_filepath)
    hd_ba = hausdorff_distance_one_direction(mesh2_filepath, mesh1_filepath)

    min_distance_bi = min(hd_ab["min_distance"], hd_ba["min_distance"])
    max_distance_bi = max(hd_ab["max_distance"], hd_ba["max_distance"])

    sm = hd_ab["mean_distance"] * hd_ab["number_points"] + hd_ba["mean_distance"] * hd_ba["number_points"]
    mean_distance_bi = sm / (hd_ab["number_points"] + hd_ba["number_points"])

    ms = (hd_ab["rms_distance"] ** 2) * hd_ab["number_points"] + (hd_ba["rms_distance"] ** 2) * hd_ba["number_points"]
    rms_distance_bi = math.sqrt(ms / (hd_ab["number_points"] + hd_ba["number_points"]))

    return {"min_distance": min_distance_bi,
            "max_distance": max_distance_bi,
            "mean_distance": mean_distance_bi,
            "rms_distance": rms_distance_bi,
            "number_points": hd_ab["number_points"]}


@numba.jit
def calculate_voxel_side_length(mesh, grid_size):
    minx, maxx = minmax(mesh.vertices[:, 0])
    miny, maxy = minmax(mesh.vertices[:, 1])
    minz, maxz = minmax(mesh.vertices[:, 2])

    return max(abs(minx - maxx) / grid_size,
               abs(miny - maxy) / grid_size,
               abs(minz - maxz) / grid_size)


@numba.jit
def _jaccard_distance(grid1, grid2):
    intersection = np.logical_and(grid1, grid2)
    intersection_count = np.count_nonzero(intersection)

    union = np.logical_or(grid1, grid2)
    union_count = np.count_nonzero(union)
    return float(intersection_count) / float(union_count)


@numba.jit
def jaccard_similarity(mesh_filepath0, mesh_filepath1, grid_size=40, exact=True):
    temp_mesh0_filepath = tempfile.mktemp(suffix=".ply")
    temp_mesh1_filepath = tempfile.mktemp(suffix=".ply")

    binvox0_filepath = temp_mesh0_filepath.replace(".ply", ".binvox")
    binvox1_filepath = temp_mesh1_filepath.replace(".ply", ".binvox")

    os.symlink(os.path.abspath(mesh_filepath0), temp_mesh0_filepath)
    os.symlink(os.path.abspath(mesh_filepath1), temp_mesh1_filepath)

    mesh0 = plyfile.PlyData.read(temp_mesh0_filepath)

    minx, maxx = minmax(mesh0['vertex']['x'])
    miny, maxy = minmax(mesh0['vertex']['y'])
    minz, maxz = minmax(mesh0['vertex']['z'])

    # -d: specify voxel grid size (default 256, max 1024)(no max when using -e)
    # -e: exact voxelization (any voxel with part of a triangle gets set)(does not use graphics card)
    # -bb <minx> <miny> <minz> <maxx> <maxy> <maxz>: force a different input model bounding box
    cmd_base = "binvox -pb "
    if exact:
        cmd_base += "-e "

    cmd_base += "-d " + str(grid_size) + " -bb " + str(minx) + " " + str(miny) + " " + str(minz) + " " + str(maxx) + " " + str(maxy) + " " + str(maxz)

    mesh0_cmd = cmd_base + " " + temp_mesh0_filepath
    mesh1_cmd = cmd_base + " " + temp_mesh1_filepath

    subprocess.call(mesh0_cmd.split(" "))
    subprocess.call(mesh1_cmd.split(" "))

    mesh0_binvox = binvox_rw.read_as_3d_array(open(binvox0_filepath, 'r'))
    mesh1_binvox = binvox_rw.read_as_3d_array(open(binvox1_filepath, 'r'))

    jaccard = _jaccard_distance(mesh0_binvox.data, mesh1_binvox.data)

    if os.path.exists(temp_mesh0_filepath):
        os.remove(temp_mesh0_filepath)

    if os.path.exists(temp_mesh1_filepath):
        os.remove(temp_mesh1_filepath)

    if os.path.exists(binvox0_filepath):
        os.remove(binvox0_filepath)

    if os.path.exists(binvox1_filepath):
        os.remove(binvox1_filepath)

    return jaccard
