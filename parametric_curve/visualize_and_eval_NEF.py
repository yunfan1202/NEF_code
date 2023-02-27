import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import random
import json
import torch
import point_cloud_utils as pcu


def visualize_pred_gt(all_pred_points, all_gt_points, name, save_fig=False, show_fig=True):
    # all_pred_points = []

    ax = plt.figure(dpi=120).add_subplot(projection='3d')

    x = [k[0] for k in all_pred_points]
    y = [k[1] for k in all_pred_points]
    z = [k[2] for k in all_pred_points]
    # print("max xyz:", max(x), max(y), max(z))
    ax.scatter(x, y, z, c='r', marker='o', s=0.5, linewidth=1, alpha=1, cmap='spectral')
    # ---------------------------------plot the gt---------------------------------
    x = [k[0] for k in all_gt_points]
    y = [k[1] for k in all_gt_points]
    z = [k[2] for k in all_gt_points]
    ax.scatter(x, y, z, c='g', marker='o', s=0.5, linewidth=1, alpha=1, cmap='spectral')

    # ax.axis('auto')
    plt.axis('off')
    # plt.xlabel("X axis")
    # plt.ylabel("Y axis")

    ax.view_init(azim=60, elev=60)
    range_size = [0, 1]
    ax.set_zlim3d(range_size[0], range_size[1])
    plt.axis([range_size[0], range_size[1], range_size[0], range_size[1]])
    if save_fig:
        plt.savefig(os.path.join(vis_dir, name + ".png"), bbox_inches='tight')
    if show_fig:
        plt.show()


def sample_points_by_grid(pred_points, num_voxels_per_axis=64):
    normals = pcu.estimate_point_cloud_normals_knn(pred_points, 16)[1]
    bbox_size = np.array([1, 1, 1])
    # The size per-axis of a single voxel
    sizeof_voxel = bbox_size / num_voxels_per_axis
    pred_sampled, _, _ = pcu.downsample_point_cloud_voxel_grid(sizeof_voxel, pred_points, normals)
    pred_sampled = pred_sampled.astype(np.float32)
    return pred_sampled


def get_pred_points(json_path, curve_type="cubic", sample_num=100):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    curves_ctl_pts = json_data['curves_ctl_pts']
    num_curves = len(curves_ctl_pts)
    print("num_curves:", num_curves)

    t = np.linspace(0, 1, sample_num)
    if curve_type == "cubic":
        # # -----------------------------------for Cubic Bezier-----------------------------------
        matrix_u = np.array([t ** 3, t ** 2, t, 1], dtype=object)
        matrix_middle = np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ])
    elif curve_type == "line":
        # # -------------------------------------for Line-----------------------------------------
        matrix_u = np.array([t, 1], dtype=object)
        matrix_middle = np.array([
            [-1, 1],
            [1, 0]
        ])

    all_points = []
    for i, each_curve in enumerate(curves_ctl_pts):
        each_curve = np.array(each_curve)  # shape: (4, 3)
        each_curve = (each_curve / 256 * 2.4) - 1.2   # based on the settings of extract point cloud

        matrix = np.matmul(np.matmul(matrix_u.T, matrix_middle), each_curve)
        for i in range(sample_num):
            all_points.append([matrix[0][i], matrix[1][i], matrix[2][i]])

    # exchange X and Y axis, do not know why yet... ...
    all_points = np.array([[pts[1], pts[0], pts[2]] for pts in all_points])
    return np.array(all_points)


def get_gt_points(name):
    base_dir = "../ABC_NEF_obj_examples"
    objs_dir = os.path.join(base_dir, "obj")
    obj_names = os.listdir(objs_dir)
    obj_names.sort()
    index_obj_names = {}
    for obj_name in obj_names:
        index_obj_names[obj_name[:8]] = obj_name
    print(index_obj_names)

    json_feats_path = os.path.join(base_dir, "chunk_0000_feats.json")
    with open(json_feats_path, 'r') as f:
        json_data_feats = json.load(f)
    json_stats_path = os.path.join(base_dir, "chunk_0000_stats.json")
    with open(json_stats_path, 'r') as f:
        json_data_stats = json.load(f)

    # get the normalize scale to help align the nerf points and gt points
    [x_min, y_min, z_min, x_max, y_max, z_max, x_range, y_range, z_range] = json_data_stats[name]["bbox"]
    scale = 1 / max(x_range, y_range, z_range)
    # print("normalize scale:", scale)
    poi_center = np.array([((x_min + x_max) / 2), ((y_min + y_max) / 2), ((z_min + z_max) / 2)]) * scale
    # print("poi:", poi_center)
    set_location = [0.5, 0.5, 0.5] - poi_center  # based on the rendering settings

    obj_path = os.path.join(objs_dir, index_obj_names[name])
    with open(obj_path, encoding='utf-8') as file:
        data = file.readlines()
    vertices_obj = [each.split(' ') for each in data if each.split(' ')[0] == 'v']
    vertices_xyz = [[float(v[1]), float(v[2]), float(v[3].replace('\n', ''))] for v in vertices_obj]

    edge_pts = []
    edge_pts_raw = []
    for each_curve in json_data_feats[name]:
        if each_curve['sharp']:
            each_edge_pts = [vertices_xyz[i] for i in each_curve['vert_indices']]
            edge_pts_raw += each_edge_pts

            gt_sampling = []
            each_edge_pts = np.array(each_edge_pts)
            for index in range(len(each_edge_pts) - 1):
                next = each_edge_pts[index + 1]
                current = each_edge_pts[index]
                num = int(np.linalg.norm(next - current) // 0.01)
                linspace = np.linspace(0, 1, num)
                gt_sampling.append(linspace[:, None] * current + (1 - linspace)[:, None] * next)
            each_edge_pts = np.concatenate(gt_sampling).tolist()
            edge_pts += each_edge_pts


    edge_pts_raw = np.array(edge_pts_raw) * scale + set_location
    edge_pts = np.array(edge_pts) * scale + set_location

    return edge_pts_raw.astype(np.float32), edge_pts.astype(np.float32)


def compute_chamfer_distance(pred_sampled, gt_points, metrics):
    chamfer_dist = pcu.chamfer_distance(pred_sampled, gt_points)
    metrics["chamfer"].append(chamfer_dist)
    print("chamfer_dist:", chamfer_dist)
    return metrics


def compute_precision_recall_IOU(pred_sampled, gt_points, metrics, thresh=0.02):
    dists_a_to_b, _ = pcu.k_nearest_neighbors(pred_sampled, gt_points,
                                              k=1)  # k closest points (in pts_b) for each point in pts_a
    correct_pred = np.sum(dists_a_to_b < thresh)
    precision = correct_pred / len(dists_a_to_b)
    metrics["precision"].append(precision)

    dists_b_to_a, _ = pcu.k_nearest_neighbors(gt_points, pred_sampled, k=1)
    correct_gt = np.sum(dists_b_to_a < thresh)
    recall = correct_gt / len(dists_b_to_a)
    metrics["recall"].append(recall)

    fscore = 2 * precision * recall / (precision + recall)
    metrics["fscore"].append(fscore)

    intersection = min(correct_pred, correct_gt)
    union = len(dists_a_to_b) + len(dists_b_to_a) - max(correct_pred, correct_gt)

    IOU = intersection / union
    metrics["IOU"].append(IOU)
    print("precision:", precision, "recall:", recall, "fscore:", fscore, "IOU:", IOU)
    return metrics


save_curve_dir = "ABC_curves_result"
vis_dir = "./visualization"
os.makedirs(vis_dir, exist_ok=True)

result_names = [each for each in os.listdir(save_curve_dir) if each.endswith("_stage2_cubic.json")]
result_names.sort()
print(len(result_names))

metrics = {
        "chamfer": [],
        "precision": [],
        "recall": [],
        "fscore": [],
        "IOU": []
    }

test_only_line = False
if test_only_line:
    with open("only_line_list.txt", 'r') as f:
        line_obj_names = f.readlines()
    line_obj_names = [each.replace('\n', '') for each in line_obj_names]
    print(line_obj_names)
    print("number of objs containing only lines:", len(line_obj_names))

for i, result_name in enumerate(result_names):       # result_name like: record_00000006_0.7_stage2_cubic.json
    name = result_name.split('_')[1]    # name like: 00000006
    if test_only_line and (name not in line_obj_names):
        continue

    print("-" * 50)
    print("processing:", i, ", name:", name)

    result_path = os.path.join(save_curve_dir, result_name)
    pred_points = get_pred_points(result_path, curve_type="cubic", sample_num=500)
    pred_sampled = sample_points_by_grid(pred_points)
    gt_points_raw, gt_points = get_gt_points(name)

    metrics = compute_chamfer_distance(pred_sampled, gt_points_raw, metrics)
    metrics = compute_precision_recall_IOU(pred_sampled, gt_points_raw, metrics, thresh=0.02)
    # print("raw preds:", pred_points.shape, ", sampled preds:", pred_sampled.shape, ", gt_raw shape:", gt_points_raw.shape, ", gt shape:", gt_points.shape)
    visualize_pred_gt(pred_points, gt_points, name, save_fig=False, show_fig=True)

for key, value in metrics.items():
    metrics[key] = round(np.mean(value), 4)
print("total CADs:", len(result_names))
print(metrics)
