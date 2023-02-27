import os
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import json
import point_cloud_utils as pcu

from ChamferDistancePytorch import fscore
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
# https://github.com/ThibaultGROUEIX/ChamferDistancePytorch


def chamfer_example():
    chamLoss = dist_chamfer_3D.chamfer_3DDist()
    points1 = torch.rand(32, 1000, 3).cuda()
    points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
    dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
    print(dist1.shape, dist2.shape, idx1.shape, idx2.shape)
    f_score, precision, recall = fscore.fscore(dist1, dist2)
    print(f_score.shape, precision.shape, recall.shape)

    chamfer_loss = torch.sqrt(dist1).mean() + torch.sqrt(dist2).mean()
    print(chamfer_loss)
    # print(idx1)
    # print(idx2)
    return 0


class Curves_Model(nn.Module):
    def __init__(self, n_curves=12, initial_params=None, curve_type="cubic"):
        super(Curves_Model, self).__init__()
        self.curve_type = curve_type
        if self.curve_type == "cubic":
            self.n_ctl_points = 4
            self.matrix_w = torch.tensor([
                [-1, 3, -3, 1],
                [3, -6, 3, 0],
                [-3, 3, 0, 0],
                [1, 0, 0, 0]
            ]).float().cuda()
        elif self.curve_type == "line":
            self.n_ctl_points = 2
            self.matrix_w = torch.tensor([
                [-1, 1],
                [1, 0]
            ]).float().cuda()
        self.matrix_t = self.get_matrix_t(num=100)

        if initial_params is None:
            params = torch.rand(n_curves, self.n_ctl_points, 3, requires_grad=True).cuda()  # n * 4 * 3
        else:
            params = initial_params.cuda()
        assert params.shape == (n_curves, self.n_ctl_points, 3)
        self.params = nn.Parameter(params)

    def initialize_params_center(self, pts_target):
        print("Initializing parameters... ...")
        init_mode = "center"
        if init_mode == "center":
            center_pts = torch.mean(pts_target.squeeze(), axis=0)
            self.params.requires_grad = False
            for i in range(len(self.params)):
                for j in range(len(self.params[i])):
                    self.params[i][j] = center_pts
            self.params.requires_grad = True

    def get_matrix_t(self, num=50):
        matrix_t = []
        if self.curve_type == "cubic":
            for t in np.linspace(0, 1, num):
                each_matrix_t = torch.tensor([
                    t * t * t,
                    t * t,
                    t,
                    1
                ])
                matrix_t.append(each_matrix_t)
        elif self.curve_type == "line":
            for t in np.linspace(0, 1, num):
                each_matrix_t = torch.tensor([
                    t,
                    1
                ])
                matrix_t.append(each_matrix_t)

        matrix_t = torch.stack(matrix_t, axis=0).float().cuda()
        return matrix_t

    def forward(self):
        matrix1 = torch.einsum('ik,kj->ij', [self.matrix_t, self.matrix_w])  # shape: [100, 4] * [4, 4] = [100, 4]
        matrix2 = torch.einsum('ik,nkj->nij', [matrix1, self.params])  # shape: [100, 4] * [n, 4, 3] = [n, 100, 3]
        pts_curve = matrix2.reshape(1, -1, 3)  # shape: [1, n * 100, 3]

        multiply = 5        # default 5
        pts_curve_m = pts_curve.repeat(1, multiply, 1)  # shape: [1, n * 100 * multiply, 3]
        noise = torch.randn_like(pts_curve_m)
        variance = 0.5      # default 0.5
        noise = (variance ** 0.5) * noise
        # print(torch.mean(noise), torch.max(noise), torch.min(noise))
        pts_curve_m = pts_curve_m + noise

        return pts_curve, pts_curve_m, self.params


def optimize_one_curve(max_iters, pts_target, alpha=5, curve_type="cubic"):
    chamLoss = dist_chamfer_3D.chamfer_3DDist()
    curve_model = Curves_Model(n_curves=1, curve_type=curve_type)
    curve_model.initialize_params_center(pts_target)
    # print(curve_model.params)

    lr = 0.5
    optimizer = torch.optim.Adam(curve_model.parameters(), lr=lr)

    for iters in range(max_iters):
        pts_curve, pts_curve_m, current_params = curve_model()

        # chamfer loss
        dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, pts_target)

        chamfer_loss_1 = alpha * torch.sqrt(dist1).mean()
        chamfer_loss_2 = torch.sqrt(dist2).mean()
        loss = chamfer_loss_1 + chamfer_loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"iters: {iters}, loss_total: {loss}, loss_1: {chamfer_loss_1}, alpha: {alpha}")
    return current_params, pts_curve, pts_curve_m


def updata_pts_target(pts_curve, pts_target):
    # delete point cloud close to current curve
    pts_curve = pts_curve.squeeze(0).cpu().detach().numpy()
    pts_target = pts_target.squeeze(0).cpu().detach().numpy()
    print("pts_curve.shape:", pts_curve.shape, "pts_target.shape:", pts_target.shape)

    distance = 4    # default 4
    dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(pts_curve, pts_target, k=100)
    delete_index = corrs_a_to_b[dists_a_to_b < distance]
    delete_index = list(set(delete_index))
    print("Deleting " + str(len(delete_index)) + " points")
    pts_target = np.delete(pts_target, delete_index, axis=0)
    # pts_target = torch.from_numpy(pts_target).float().unsqueeze(0).cuda()

    return pts_target, len(delete_index)


def Line2Cubic(curves_ctl_pts):
    curves_ctl_pts_new = []
    for each_curve in curves_ctl_pts:
        each_curve = np.array(each_curve)
        extra_pts1 = 2 / 3 * each_curve[0] + 1 / 3 * each_curve[1]
        extra_pts2 = 1 / 3 * each_curve[0] + 2 / 3 * each_curve[1]
        new_curve = np.array([each_curve[0], extra_pts1, extra_pts2, each_curve[1]]).tolist()
        curves_ctl_pts_new.append(new_curve)
    return curves_ctl_pts_new


if __name__ == '__main__':
    # chamfer_example()
    print("->Loading Point Cloud... ...")
    point_cloud_dir = "ABC_point_clouds"
    save_curve_dir = "ABC_curves_result"
    os.makedirs(save_curve_dir, exist_ok=True)

    scene_names = os.listdir(point_cloud_dir)
    scene_names.sort()

    for i, scene_name in enumerate(scene_names):
        print("-" * 50)
        print("processing:", i, ", scene_name:", scene_name)
        pcd_path = os.path.join(point_cloud_dir, scene_name)
        pcd = o3d.io.read_point_cloud(pcd_path)
        init_pts_target = torch.from_numpy(np.asarray(pcd.points)).float().unsqueeze(0).cuda()
        print("initial pts_target.shape:", init_pts_target.shape)
        curve_type = "line"

        # # stage 1: ------------------------per curve optimization-------------------------
        print("Ready to conduct stage 1 ... ...")
        start = time.perf_counter()
        pts_target = init_pts_target.clone()
        cur_curves = []
        # total_curves = 12
        for i in range(100):
            print('=' * 70)
            curves_params, pts_curve, pts_curve_m = optimize_one_curve(max_iters=400, pts_target=pts_target, alpha=5, curve_type=curve_type)
            pts_target, delete_num = updata_pts_target(pts_curve, pts_target)
            if delete_num > 20:
                cur_curves.append(np.array(curves_params.detach().cpu()))
            print("Current pts_target.shape:", pts_target.shape)
            print('Current number of Curves:', len(cur_curves))
            if pts_target.shape[0] < 20:   # if there are very few points, stop the optimazation
                break
            pts_target = torch.from_numpy(pts_target).float().unsqueeze(0).cuda()

        cur_curves = np.array(cur_curves).squeeze(1)    # (total_curves, 1, 4, 3) to (total_curves, 4, 3)
        print("total curves:", cur_curves.shape)
        print("Total time comsumed", time.perf_counter() - start)

        json_data = {
            "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "scene_name": scene_name,
            'curves_ctl_pts': cur_curves.tolist()
        }
        file_name = "record_" + scene_name[:-4] + "_stage1_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        print("json file saved in", json_path)

        # stage 2:------------------------all curve refinement-------------------------
        print("Ready to conduct stage 2 ... ...")
        time.sleep(1)
        file_name = "record_" + scene_name[:-4] + "_stage1_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        curves_ctl_pts = json_data['curves_ctl_pts']
        # print(curves_ctl_pts)
        # print(torch.tensor(curves_ctl_pts).shape)
        print("Number of curves:", len(curves_ctl_pts))

        start = time.perf_counter()
        chamLoss = dist_chamfer_3D.chamfer_3DDist()

        curve_type = "cubic"
        curves_ctl_pts_new = Line2Cubic(curves_ctl_pts)
        curve_model = Curves_Model(n_curves=len(curves_ctl_pts), initial_params=torch.tensor(curves_ctl_pts_new), curve_type=curve_type)

        # print(curve_model.params)

        lr = 0.5
        optimizer = torch.optim.Adam(curve_model.parameters(), lr=lr)
        for iters in range(1000):
            pts_curve, pts_curve_m, current_params = curve_model()

            # calculate endpoints loss
            end_pts = torch.stack([current_params[:, 0, :], current_params[:, -1, :]], dim=1).reshape(-1, 3)  # torch.Size([n * 2, 3])
            dists = torch.pdist(end_pts, p=2)       # torch.Size([(n * 2) ** 2 / 2 - n])
            mask = torch.ones_like(dists)
            mask[dists > 4] = 0     # default 4
            masked_dists = dists * mask
            loss_end_pts = 0.01 * masked_dists.sum()

            # chamfer loss
            dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, init_pts_target)

            alpha = 1
            chamfer_loss_1 = alpha * torch.sqrt(dist1).mean()
            chamfer_loss_2 = torch.sqrt(dist2).mean()
            loss = chamfer_loss_1 + chamfer_loss_2 + loss_end_pts

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iters + 1) % 100 == 0:
                print(f"iters: {iters}, loss_total: {loss}, loss_1: {chamfer_loss_1}, loss_end_pts: {loss_end_pts}, alpha: {alpha}")

        print("Stage 2 time comsumed", time.perf_counter() - start)


        json_data = {
            "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "scene_name": scene_name,
            'curves_ctl_pts': current_params.tolist()
        }
        file_name = "record_" + scene_name[:-4] + "_stage2_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        print("json file saved in", json_path)

