import cPickle
import ctypes
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pcl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN


def point_inline_check(x, y, image_width, image_height):
    return 0 < x < image_height - 1 and 0 < y < image_width - 1


def find_table(
        cloud  # type: pcl.PointCloud
):
    seg = cloud.make_segmenter()
    # Optional
    seg.set_optimize_coefficients(True)
    # Mandatory
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.1)
    # seg.set_max_iterations(1000)

    nr_points = cloud.size
    min_distance = 10000.0
    # table_pcd = np.empty()

    inliers, model = seg.segment()

    print(inliers)
    print(model)

    # while cloud

    import IPython
    IPython.embed()


def find_table_plane(pcd):
    lib = np.ctypeslib.load_library("table_finder/testlib", ".")
    array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
    out = np.zeros(500000)
    pcd_arr = np.ones((pcd.shape[0] * 3))
    pcd_arr[0:pcd.shape[0]] = pcd[:, 0]
    pcd_arr[pcd.shape[0]:pcd.shape[0] * 2] = pcd[:, 1]
    pcd_arr[2 * pcd.shape[0]:3 * pcd.shape[0]] = pcd[:, 2]

    lib.add_one.restype = ctypes.c_int
    lib.add_one.argtypes = [array_1d_double, ctypes.c_int, ctypes.c_int, array_1d_double]
    num_idx = lib.add_one(pcd_arr, pcd.shape[0], 3, out)

    res_out = np.zeros((num_idx, 3))
    res_out[:, 0] = out[0:num_idx]
    res_out[:, 1] = out[num_idx:2 * num_idx]
    res_out[:, 2] = out[2 * num_idx:3 * num_idx]

    return res_out


def scan_line(mask, target_idx, seed_coord, image_width, image_height):
    seed_idx = 0
    mask_cp = np.copy(mask)
    stack = [seed_coord]
    while len(stack) > 0:
        coord = stack.pop()
        x = coord[0]
        y = coord[1]
        x1 = x

        while point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y] == seed_idx: x1 -= 1
        x1 += 1

        span_above = span_below = False
        while point_inline_check(x1, y) and mask_cp[x1, y] == seed_idx:
            mask_cp[x1, y] = target_idx
            if not span_above and point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y - 1] == seed_idx:
                # print "append"
                stack.append((x1, y - 1))
                span_above = True
            elif span_above and point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y - 1] != seed_idx:
                # print "no append"
                span_above = False

            if not span_below and point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y + 1] == seed_idx:
                # print "append"
                stack.append((x1, y + 1))
                span_below = True
            elif span_below and point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y + 1] != seed_idx:
                # print "no append"
                span_below = False
            x1 += 1

    return mask_cp


def search_around_point(idx_list, mask, image_width, image_height):
    mask_updated = np.copy(mask)
    fill_idx = 3
    for idx in idx_list:
        if mask_updated[idx[0], idx[1]] == 0:
            mask_updated = scan_line(mask_updated, fill_idx, idx, image_width, image_height)
            fill_idx += 1

    return mask_updated, fill_idx


def validate_bg_points(idx_0_table_mask, table_mask, table_full_mask, image_width, image_height):
    idx_0_table_mask_ls = []
    for i in xrange(idx_0_table_mask[0].shape[0]):
        x = idx_0_table_mask[0][i]
        y = idx_0_table_mask[1][i]
        if (1 in table_mask[0:x, y] or 2 in table_mask[0:x, y]) and \
                (1 in table_mask[x:image_height, y] or 2 in table_mask[x:image_height, y]) \
                and (1 in table_mask[x, 0:y] or 1 in table_mask[x, 0:y]) \
                and (1 in table_mask[x, y:image_width] or 2 in table_mask[x, y:image_width]):

            if not np.any(table_mask[x:x + 5, y]) and not np.any(table_mask[x:x - 5, y]) \
                    and not np.any(table_mask[x, y:y + 5]) and not np.any(table_mask[x, y - 5]):
                if table_full_mask[x, y] == 1:
                    idx_0_table_mask_ls.append((x, y))

    return idx_0_table_mask_ls


def clustering_2d(cam_model, table_mask, table_top_pcd):
    raise NotImplementedError("clustering_2d not yet working")
    #
    # table_top_mask = cam_model.pcl_to_2dcoord(table_top_pcd)
    # table_full_mask = np.logical_or(table_top_mask, table_mask)
    # _, contours, _ = cv2.findContours(table_full_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(table_mask, contours, -1, (2), 3)
    #
    # idx_0_table_mask = np.where(table_mask == 0)
    # idx_0_table_mask_ls = validate_bg_points(idx_0_table_mask, table_mask, table_full_mask)
    #
    # return search_around_point(idx_0_table_mask_ls, table_mask)


def clean_tabletop_pcd(cam_model, table_mask, tabletop_pcd):
    tabletop_pcd_clean = np.zeros(tabletop_pcd.shape)
    index = 0
    for i in xrange(tabletop_pcd.shape[0]):
        coord_2d = cam_model.project_3d_to_2d(tabletop_pcd[i, :])
        coord_2d = list(coord_2d)
        coord_2d[0] = int(round(coord_2d[0]))
        coord_2d[1] = int(round(coord_2d[1]))
        if table_mask[coord_2d[1], coord_2d[0]] == 0:
            tabletop_pcd_clean[index, :] = tabletop_pcd[i, :]
            index += 1

    return tabletop_pcd_clean[:index, :]


def density_clustering(table_top_pcd_clean):
    return DBSCAN(eps=0.1, min_samples=1000).fit(table_top_pcd_clean)


def seg_pcl_from_labels(cluster_labels, table_top_pcd_clean):
    obj_3d = []
    for i in range(np.unique(cluster_labels).shape[0] - 1):
        idx = np.where(cluster_labels == i)
        obj_i = table_top_pcd_clean[idx, :]
        obj_i = np.squeeze(obj_i, axis=0)
        obj_3d.append(obj_i)

    return obj_3d


def clustering_3d(cam_model, table_mask, table_top_pcd):
    table_top_pcd_clean = clean_tabletop_pcd(cam_model, table_mask, table_top_pcd)

    db = density_clustering(table_top_pcd_clean)
    cluster_labels = db.labels_
    obj_pcl = seg_pcl_from_labels(cluster_labels, table_top_pcd_clean)
    return obj_pcl


def show_img_with_mask(img, mask):
    img[:, :, 0] = np.multiply(img[:, :, 0], mask)
    img[:, :, 1] = np.multiply(img[:, :, 1], mask)
    img[:, :, 2] = np.multiply(img[:, :, 2], mask)

    plt.imshow(img)
    plt.show()


def show_3d_plot(pcd):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # type: Axes3D

    ax.scatter(pcd[:, :, 0].flatten(), pcd[:, :, 1].flatten(), pcd[:, :, 2].flatten())

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def pcd_max_min_dist_viewer(pcd):
    print "min x, max x: ", np.min(pcd[:, 0]), np.max(pcd[:, 0])
    print "min y, max y: ", np.min(pcd[:, 1]), np.max(pcd[:, 1])
    print "min z, max z: ", np.min(pcd[:, 2]), np.max(pcd[:, 2])
    print "min distance, max distance: ", np.min(np.square(pcd[:, 0]) + np.square(pcd[:, 1]) + np.square(pcd[:, 2])), \
        np.max(np.square(pcd[:, 0]) + np.square(pcd[:, 1]) + np.square(pcd[:, 2]))


class Transformer:
    def __init__(self, camera_info, image_width, image_height, fx=0, fy=0, cx=0, cy=0, camera_info_object=None):
        if camera_info:
            self.fx = int(camera_info_object.K[0])
            self.fy = int(camera_info_object.K[4])
            self.cx = int(camera_info_object.K[2])
            self.cy = int(camera_info_object.K[5])
        else:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy

        self.image_height = image_height
        self.image_width = image_width

    def pcl_to_2dcoord(self, pcd):
        mask = np.zeros((self.image_height, self.image_width))
        for i in xrange(pcd.shape[0]):
            xy = self.project_3d_to_2d(pcd[i, :])
            mask[xy[1], xy[0]] = 1
        return mask

    def pixel_to_pcl(self, uv_pixel):
        unit_pcl = np.zeros((len(uv_pixel), 3))

        for key, pixel in enumerate(uv_pixel):
            unit_pcl[key, :] = self.project_2d_to_3d(pixel)
        return unit_pcl

    def project_3d_to_2d(self, pt):
        x = int((self.fx * pt[0]) / pt[2] + self.cx)
        y = int((self.fy * pt[1]) / pt[2] + self.cy)
        return x, y

    def project_2d_to_3d(self, px):
        unit_pcl = np.zeros(3)
        unit_pcl[0] = (px[0] - self.cx) / self.fx
        unit_pcl[1] = (px[1] - self.cy) / self.fy
        unit_pcl[2] = 1.0

        return unit_pcl


def save_pcl(pcd, name):
    pc = pcl.PointCloud()
    pc.from_array(pcd.astype(np.float32))
    pc.to_file(name)


def table_plane_pcl(p, path):
    table_pcd = []
    min_distance = 10000000000000
    # p = pcl.PointCloud()
    for f in os.listdir(path):
        p.from_file(os.path.join(path, f))
        pcd = p.to_array()
        dist = np.min(np.square(pcd[:, 0]) + np.square(pcd[:, 1]) + np.square(pcd[:, 2]))
        if dist < min_distance:
            min_distance = dist
            table_pcd = pcd

    return table_pcd


def find_normal_vector(pcd):
    rows = pcd.shape[0]
    normal_vector = 0
    max_itr = 20
    test_vector = np.zeros((3, 1))

    for itr in xrange(0, max_itr):
        idx = random.sample(range(0, rows), 3)
        point1 = pcd[idx[0], :]
        point2 = pcd[idx[1], :]
        point3 = pcd[idx[2], :]

        vector1 = point2 - point1
        vector2 = point3 - point1

        test_vector = -point1

        normal_vector += np.cross(vector1, vector2)

    if np.dot(normal_vector, test_vector) < 0:
        return -normal_vector / max_itr
    else:
        return normal_vector / max_itr


def pcl_above_plane(plane_pcl, all_pcl):
    anchor_point = np.mean(plane_pcl, axis=0)

    roi_pcl = np.zeros(all_pcl.shape)
    index = 0

    normal_vector = find_normal_vector(plane_pcl)

    for i in xrange(all_pcl.shape[0]):
        if np.dot(normal_vector, all_pcl[i, :] - anchor_point) > 0:
            roi_pcl[index, :] = all_pcl[i, :]
            index += 1
    return roi_pcl[:index, :]


def pcl_clustering(pcd, image_width, image_height, clustering_dim='3D', **kwargs):
    if ("fx" in kwargs and "fy" in kwargs and "cx" in kwargs and "cy" in kwargs and "camera_info" in kwargs) or \
       ("camera_info" in kwargs and "camera_info_object" in kwargs):
        camera_model = Transformer(image_width, image_height, **kwargs)
    else:
        raise ValueError("Keyword arguments must be (fx, fy, cx, cy, camera_info) or (camera_info, camera_info_object)")

    table_pcd = find_table_plane(pcd)
    table_top_pcd = pcl_above_plane(table_pcd, pcd)
    table_mask = camera_model.pcl_to_2dcoord(table_pcd)

    if clustering_dim == '2D':  # not available yet
        return clustering_2d(camera_model, table_mask, table_top_pcd)
    if clustering_dim == '3D':
        return clustering_3d(camera_model, table_mask, table_top_pcd)


def __main():
    camera_info = True  # True if you use camInfo object, False if you use fx,fy, cx,cy
    input_example_path_base = './example/input/'
    output_example_path_base = './example/output/'

    pcd_path_input = os.path.join(input_example_path_base, 'example_pcl.pcd')
    camera_info_path_output = os.path.join(input_example_path_base, 'example_pkl.pkl')

    image_width = 640
    image_height = 480

    p = pcl.PointCloud()
    p.from_file(pcd_path_input)
    input_pcd = p.to_array()
    input_pcd = input_pcd[~np.isnan(input_pcd[:, 2]), :]

    with open(camera_info_path_output, 'rb') as f:
        camera_info_object = cPickle.load(f)

    if camera_info:
        output = pcl_clustering(input_pcd, image_width, image_height, camera_info_object=camera_info_object,
                                camera_info=camera_info)
    else:
        output = pcl_clustering(input_pcd, image_width, image_height, fx=int(camera_info_object.K[0]),
                                fy=int(camera_info_object.K[4]), cx=int(camera_info_object.K[2]),
                                cy=int(camera_info_object.K[5]), camera_info=camera_info)

    for i in xrange(len(output)):
        pcd_path_output = os.path.join(output_example_path_base, 'example_pcl_output_{:02d}.pcd'.format(i))
        p.from_array(output[i].astype(np.float32))
        p.to_file(pcd_path_output)


if __name__ == "__main__":
    __main()
