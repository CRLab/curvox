import curvox
from curvox import time_fn

meshfilename1 = 'resources/unsmoothed.ply'
meshfilename2 = 'resources/smoothed.ply'


def test_hausdorff_distance_one_direction():

    time_fn(curvox.mesh_comparisons.hausdorff_distance_one_direction, meshfilename1, meshfilename2)
    distance_uni = time_fn(curvox.mesh_comparisons.hausdorff_distance_one_direction, meshfilename1, meshfilename2)

    curvox.mesh_comparisons.print_hausdorff(distance_uni)


def test_hausdorff_distance_bi():

    time_fn(curvox.mesh_comparisons.hausdorff_distance_bi, meshfilename1, meshfilename2)
    distance_bi = time_fn(curvox.mesh_comparisons.hausdorff_distance_bi, meshfilename1, meshfilename2)

    curvox.mesh_comparisons.print_hausdorff(distance_bi)


def test_jaccard_similarity():

    time_fn(curvox.mesh_comparisons.jaccard_similarity, meshfilename1, meshfilename2)
    distance_j = time_fn(curvox.mesh_comparisons.jaccard_similarity, meshfilename1, meshfilename2)

    print(distance_j)
