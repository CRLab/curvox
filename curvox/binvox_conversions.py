import numpy as np

def binvox_to_pcl(vox):
    """
    Takines a binvox_rw.binvox_rw.Voxels, and converts it to a nx3 numpy array
    representing the occupied voxels as a pointcloud.
    :type vox: binvox_rw.binvox_rw.Voxels
    :rtype numpy.ndarray
    """

    # get indices for occupied voxels:
    # pts.shape == (n, 3) 
    pts = np.array(vox.data.nonzero()).T
    
    # single_voxel_dim is the dimensions in meters of a single voxel
    # single_voxel_dim == array([ 0.00092833,  0.00092833,  0.00092833])
    single_voxel_dim = vox.scale / np.array(vox.dims)
    
    # convert pts back to meters:
    pts_offset = pts * single_voxel_dim
    
    # translate the points back to their original position
    pts = pts_offset + vox.translate

    return pts
