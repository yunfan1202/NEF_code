import torch
from torch import nn


class L1Loss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        targets = targets.squeeze(0)

        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
        return self.coef * loss


class MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        targets = targets.squeeze(0)

        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
        return self.coef * loss


class RGB_density_consistency(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs):
        rgbs_c = inputs['spacial_rgbs_coarse'].view(-1, 1)
        sigmas_c = inputs['edge_sigmas_coarse'].view(-1, 1)
        loss_total = self.loss(rgbs_c, sigmas_c)
        if 'spacial_rgbs_fine' in inputs and 'edge_sigmas_fine' in inputs:
            rgbs_f = inputs['spacial_rgbs_fine'].view(-1, 1)
            sigmas_f = inputs['edge_sigmas_fine'].view(-1, 1)
            loss_total += self.loss(rgbs_f, sigmas_f)
        return self.coef * loss_total


class Adaptive_MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='none')

    # def get_mask_mse(self, rgbs_tensor):
    #     # print(torch.max(rgbs_tensor), torch.min(rgbs_tensor), torch.mean(rgbs_tensor))
    #     thresh_big = 0.7
    #     thresh_small = 0.3
    #     num_positive = (torch.sum(rgbs_tensor > thresh_big) + 1).float()    # +1 to avoid 0 (no gradient)
    #     num_negative = (torch.sum(rgbs_tensor < thresh_small) + 1).float()
    #     # print("num_positive:", num_positive, "num_negative:", num_negative)
    #     mask = torch.zeros_like(rgbs_tensor)
    #     beta = 1.0
    #     mask[rgbs_tensor > thresh_big] = 1.0 * num_negative / (num_positive + num_negative)
    #     mask[rgbs_tensor < thresh_small] = beta * num_positive / (num_positive + num_negative)
    #     # print(mask, mask.shape)
    #     return mask

    def get_mask_mse(self, rgbs_tensor):
        # print(torch.max(rgbs_tensor), torch.min(rgbs_tensor), torch.mean(rgbs_tensor))
        thresh = 0.3
        num_positive = (torch.sum(rgbs_tensor > thresh)).float()    # +1 to avoid 0 (no gradient)
        num_negative = (torch.sum(rgbs_tensor <= thresh)).float()
        # print("num_positive:", num_positive, "num_negative:", num_negative)
        mask = torch.zeros_like(rgbs_tensor)

        mask[rgbs_tensor > thresh] = 1.0 * (num_negative + 1) / (num_positive + num_negative)
        mask[rgbs_tensor <= thresh] = 1.0 * (num_positive + 1) / (num_positive + num_negative)
        # print(mask, mask.shape)
        return mask

    def forward(self, inputs, targets):
        mask = self.get_mask_mse(targets)

        targets = targets.squeeze(0)
        loss_coarse = self.loss(inputs['rgb_coarse'], targets)
        loss_total = (loss_coarse * mask).mean()

        if 'rgb_fine' in inputs:
            loss_fine = self.loss(inputs['rgb_fine'], targets)
            loss_total += (loss_fine * mask).mean()
        return self.coef * loss_total


class Sparsity_Loss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
    # def get_mask_ray(self, rgbs_tensor):
    #     # print(torch.max(rgbs_tensor), torch.min(rgbs_tensor), torch.mean(rgbs_tensor))
    #     thresh = 0.1
    #     mask = torch.zeros_like(rgbs_tensor)
    #     mask[rgbs_tensor < thresh] = 1
    #     # print(mask, mask.shape)
    #     return mask

    def get_mask_ray(self, rgbs_tensor):
        mask = torch.zeros_like(rgbs_tensor)
        # mask[rgbs_tensor == 0] = 1
        mask[rgbs_tensor <= 0.3] = 1
        # print(mask, mask.shape)
        return mask

    def forward(self, inputs, rgbs):
        mask = self.get_mask_ray(rgbs)

        # print(mask.shape)     # batch_size * 1
        # print(inputs['sigmas_coarse'].shape)  # batch_size * n_samples  (1024 * 64)
        # print(inputs['sigmas_fine'].shape)

        # sigmas_c = inputs['sigmas_coarse']
        sigmas_c = inputs['edge_sigmas_coarse']
        mask_c = mask.repeat(1, sigmas_c.shape[1])  # # batch_size * n_samples  (1024 * 64)
        sigmas_c = sigmas_c.view(-1, 1)
        mask_c = mask_c.view(-1, 1)
        loss_coarse = torch.log(1 + torch.square(sigmas_c) / 0.5)
        loss_total = (loss_coarse * mask_c).mean()

        # if 'sigmas_fine' in inputs:
        #     sigmas_f = inputs['sigmas_fine']
        if 'edge_sigmas_fine' in inputs:
            sigmas_f = inputs['edge_sigmas_fine']
            mask_f = mask.repeat(1, sigmas_f.shape[1])
            sigmas_f = sigmas_f.view(-1, 1)
            mask_f = mask_f.view(-1, 1)
            loss_fine = torch.log(1 + torch.square(sigmas_f) / 0.5)
            loss_total += (loss_fine * mask_f).mean()

        return self.coef * loss_total


loss_dict = {'l1': L1Loss,
             'mse': MSELoss,
             'rgb_density_consistency': RGB_density_consistency,
             'adaptive_mse': Adaptive_MSELoss,
             'sparsity': Sparsity_Loss}
