import numba
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pcl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN


@numba.jit
def point_inline_check(x, y, image_width, image_height):
    return 0 < x < image_height - 1 and 0 < y < image_width - 1


def find_table(cloud, min_distance=10000.0, distance_threshold=0.05):
    """
    :param cloud: Input scene cloud
    :type cloud: pcl.PointCloud
    :param min_distance: Minimum distance from camera to plane
    :type min_distance: float
    :param distance_threshold: Distance from camera frame threshold
    :type distance_threshold: float
    :return: Pointcloud of table in input cloud. Returns an empty cloud if no table was found
    :rtype: pcl.PointCloud
    """
    # the total number of points in the cloud
    nr_points = cloud.size

    table = pcl.PointCloud()

    # loop through the pointcloud and get the planes iteratively
    while cloud.size > 0.3 * nr_points:
        # not sure why normals here
        # seg = cloud.make_segmenter_normals()
        seg = cloud.make_segmenter()
        # Optional
        seg.set_optimize_coefficients(True)
        # Mandatory
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        # seg.set_max_iterations(1000)
        seg.set_distance_threshold(distance_threshold)
        indices, model = seg.segment()
        if len(indices) == 0:
            print("Could not estimate a planar model for the given dataset.")
            exit(0)
        cloud_plane = cloud.extract(indices, negative=False)
        cp_matrix = np.asarray(cloud_plane)

        # calculate the norm for all the points and get the smallest one (i.e shortest distance)
        temp_min_distance = np.amin(np.linalg.norm(cp_matrix, axis=1))
        if temp_min_distance < min_distance:
            table = cloud_plane
            min_distance = temp_min_distance

        # extract out the rest of the points
        cloud = cloud.extract(indices, negative=True)

    return table


@numba.jit
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
            if not span_above and point_inline_check(x1, y, image_width, image_height) and mask_cp[
                x1, y - 1] == seed_idx:
                # print "append"
                stack.append((x1, y - 1))
                span_above = True
            elif span_above and point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y - 1] != seed_idx:
                # print "no append"
                span_above = False

            if not span_below and point_inline_check(x1, y, image_width, image_height) and mask_cp[
                x1, y + 1] == seed_idx:
                # print "append"
                stack.append((x1, y + 1))
                span_below = True
            elif span_below and point_inline_check(x1, y, image_width, image_height) and mask_cp[x1, y + 1] != seed_idx:
                # print "no append"
                span_below = False
            x1 += 1

    return mask_cp


@numba.jit
def search_around_point(idx_list, mask, image_width, image_height):
    mask_updated = np.copy(mask)
    fill_idx = 3
    for idx in idx_list:
        if mask_updated[idx[0], idx[1]] == 0:
            mask_updated = scan_line(mask_updated, fill_idx, idx, image_width, image_height)
            fill_idx += 1

    return mask_updated, fill_idx


@numba.jit
def validate_bg_points(idx_0_table_mask, table_mask, table_full_mask, image_width, image_height):
    idx_0_table_mask_ls = []
    for i in range(idx_0_table_mask[0].shape[0]):
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


@numba.jit
def clean_tabletop_pcd(cam_model, table_mask, tabletop_pcd):
    tabletop_pcd_clean = np.zeros(tabletop_pcd.shape)
    index = 0
    for i in range(tabletop_pcd.shape[0]):
        coord_2d = cam_model.project_3d_to_2d(tabletop_pcd[i, :])
        coord_2d = list(coord_2d)
        coord_2d[0] = int(round(coord_2d[0])) - 1
        coord_2d[1] = int(round(coord_2d[1])) - 1
        if table_mask[coord_2d[1], coord_2d[0]] == 0:
            tabletop_pcd_clean[index, :] = tabletop_pcd[i, :]
            index += 1

    return tabletop_pcd_clean[:index, :]


@numba.jit
def density_clustering(table_top_pcd_clean):
    return DBSCAN(eps=0.1, min_samples=1000).fit(table_top_pcd_clean)


@numba.jit
def seg_pcl_from_labels(cluster_labels, table_top_pcd_clean):
    obj_3d = []
    for i in range(np.unique(cluster_labels).shape[0] - 1):
        idx = np.where(cluster_labels == i)
        obj_i = table_top_pcd_clean[idx, :]
        obj_i = np.squeeze(obj_i, axis=0)
        obj_3d.append(obj_i)

    return obj_3d


@numba.jit
def clustering_3d(cam_model, table_mask, table_top_pcd):
    table_top_pcd_clean = clean_tabletop_pcd(cam_model, table_mask, table_top_pcd)

    db = density_clustering(table_top_pcd_clean)
    cluster_labels = db.labels_
    obj_pcl = seg_pcl_from_labels(cluster_labels, table_top_pcd_clean)
    return obj_pcl


@numba.jit
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
    print("min x, max x: ", np.min(pcd[:, 0]), np.max(pcd[:, 0]))
    print("min y, max y: ", np.min(pcd[:, 1]), np.max(pcd[:, 1]))
    print("min z, max z: ", np.min(pcd[:, 2]), np.max(pcd[:, 2]))
    print("min distance, max distance: ", np.min(np.square(pcd[:, 0]) + np.square(pcd[:, 1]) + np.square(pcd[:, 2])),
          np.max(np.square(pcd[:, 0]) + np.square(pcd[:, 1]) + np.square(pcd[:, 2])))


class Transformer:
    def __init__(self, image_width, image_height, fx=0, fy=0, cx=0, cy=0, camera_info_object=None):
        if camera_info_object:
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

    @numba.jit
    def pcl_to_2dcoord(self, pcd):
        mask = np.zeros((self.image_height, self.image_width))
        for i in range(pcd.shape[0]):
            xy = self.project_3d_to_2d(pcd[i, :])
            mask[xy[1], xy[0]] = 1
        return mask

    @numba.jit
    def pixel_to_pcl(self, uv_pixel):
        unit_pcl = np.zeros((len(uv_pixel), 3))

        for key, pixel in enumerate(uv_pixel):
            unit_pcl[key, :] = self.project_2d_to_3d(pixel)
        return unit_pcl

    @numba.jit
    def project_3d_to_2d(self, pt):
        x = int((self.fx * pt[0]) / pt[2] + self.cx)
        y = int((self.fy * pt[1]) / pt[2] + self.cy)
        return x, y

    @numba.jit
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


@numba.jit
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


@numba.jit
def find_normal_vector(pcd):
    rows = pcd.shape[0]
    normal_vector = 0
    max_itr = 20
    test_vector = np.zeros((3, 1))

    for _ in range(0, max_itr):
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


@numba.jit
def pcl_above_plane(plane_pcl, all_pcl):
    anchor_point = np.mean(plane_pcl, axis=0)

    roi_pcl = np.zeros(all_pcl.shape)
    index = 0

    normal_vector = find_normal_vector(plane_pcl)

    for i in range(all_pcl.shape[0]):
        if np.dot(normal_vector, all_pcl[i, :] - anchor_point) > 0:
            roi_pcl[index, :] = all_pcl[i, :]
            index += 1
    return roi_pcl[:index, :]


@numba.jit
def pcd_filter(input_cloud, x_min, x_max, y_min, y_max, z_min, z_max):
    if isinstance(input_cloud, pcl.PointCloud):
        cloud_np = input_cloud.to_array()
    elif isinstance(input_cloud, np.ndarray):
        cloud_np = input_cloud
    else:
        raise ValueError("input_cloud must be pcl.PointCloud or np.ndarray")

    ll = np.array([x_min, y_min, z_min])  # lower-left
    ur = np.array([x_max, y_max, z_max])  # upper-right

    inidx = np.all(np.logical_and(ll <= cloud_np, cloud_np <= ur), axis=1)
    inner_points = cloud_np[inidx]
    # outbox = pts[np.logical_not(inidx)]

    out_cloud = pcl.PointCloud()
    out_cloud.from_array(inner_points)
    return out_cloud


def pcl_clustering(input_cloud, image_width, image_height, clustering_dim='3D', distance_threshold=0.05, **kwargs):
    """

    :param input_cloud:
    :param image_width:
    :param image_height:
    :param clustering_dim:
    :param distance_threshold:
    :param kwargs:
    :return:
    """
    if ("fx" in kwargs and "fy" in kwargs and "cx" in kwargs and "cy" in kwargs) or ("camera_info_object" in kwargs):
        camera_model = Transformer(image_width, image_height, **kwargs)
    else:
        raise ValueError("Keyword arguments must be (fx, fy, cx, cy) or (camera_info_object)")

    table_pcd = find_table(input_cloud, distance_threshold=distance_threshold)
    table_np = table_pcd.to_array()
    input_cloud_np = input_cloud.to_array()
    table_top_pcd = pcl_above_plane(table_np, input_cloud_np)
    table_mask = camera_model.pcl_to_2dcoord(table_np)

    if clustering_dim == '2D':  # not available yet
        return clustering_2d(camera_model, table_mask, table_top_pcd), table_pcd
    if clustering_dim == '3D':
        return clustering_3d(camera_model, table_mask, table_top_pcd), table_pcd
