import pcl


def binvox_to_np(binvox_filename):
    ''' Read in a binvox file and return a RLE inside a numpy array '''
    pass


def np_to_binvox(arr):
    ''' Write binvox file via numpy array containing RLE '''
    pass


def pcd_to_np(pcd_filename):
    ''' Read in PCD then return nx3 numpy array '''
    pcd = pcl.load(pcd_filename)
    return pcl_to_np(pcd)


def pcl_to_np(pointcloud):
    ''' Convert PCL pointcloud to numpy nx3 numpy array '''
    # gt_np shape is (#pts, 3)
    pcd_np = pointcloud.to_array()
    return pcd_np


def np_to_pcl(arr):
    ''' Convert nx3 numpy array to PCL pointcloud '''
    new_pcd = pcl.PointCloud()
    new_pcd.from_array(arr)
    return new_pcd