import torch
import numpy as np
from ._base import iona_register_transform, iona_get_CB_positions


@iona_register_transform("corrupt_chi_angle")
class IONACorruptChiAngle(object):

    def __init__(
        self, iona_ratio_mask=0.1, iona_add_noise=True, iona_maskable_flag_attr=None
    ):
        super().__init__()
        self.iona_ratio_mask = iona_ratio_mask
        self.iona_add_noise = iona_add_noise
        self.iona_maskable_flag_attr = iona_maskable_flag_attr

    def iona_normalize_angles(self, iona_angles):
        iona_angles = iona_angles % (2 * np.pi)
        return torch.where(iona_angles > np.pi, iona_angles - 2 * np.pi, iona_angles)

    def iona_get_min_dist(self, iona_data, iona_center_idx):
        iona_pos_beta_all = iona_get_CB_positions(
            iona_data["pos_atoms"], iona_data["mask_atoms"]
        )
        iona_pos_beta_center = iona_pos_beta_all[iona_center_idx]
        iona_cdist = torch.cdist(iona_pos_beta_all, iona_pos_beta_center)
        iona_min_dist = iona_cdist.min(dim=1)[0]
        return iona_min_dist

    def iona_get_noise_std(self, iona_min_dist):
        return torch.clamp_min(-1 / 16 * iona_min_dist + 1, 0)

    def iona_get_flip_prob(self, iona_min_dist):
        return torch.where(
            iona_min_dist <= 8.0,
            torch.full_like(iona_min_dist, 0.25),
            torch.zeros_like(iona_min_dist),
        )

    def iona_add_chi_gaussian_noise(self, iona_chi, iona_noise_std, iona_chi_mask):
        iona_noise = (
            torch.randn_like(iona_chi) * iona_noise_std[:, None] * iona_chi_mask
        )
        return self.iona_normalize_angles(iona_chi + iona_noise)

    def iona_random_flip_chi(self, iona_chi, iona_flip_prob, iona_chi_mask):
        iona_delta = (
            torch.where(
                torch.rand_like(iona_chi) <= iona_flip_prob[:, None],
                torch.full_like(iona_chi, np.pi),
                torch.zeros_like(iona_chi),
            )
            * iona_chi_mask
        )
        return self.iona_normalize_angles(iona_chi + iona_delta)

    def __call__(self, iona_data):
        iona_L = iona_data["aa"].size(0)
        iona_idx = torch.arange(0, iona_L)
        iona_num_mask = max(int(self.iona_ratio_mask * iona_L), 1)
        if self.iona_maskable_flag_attr is not None:
            iona_flag = iona_data[self.iona_maskable_flag_attr]
            iona_idx = iona_idx[iona_flag]
        iona_idx = iona_idx.tolist()
        np.random.shuffle(iona_idx)
        iona_idx_mask = iona_idx[:iona_num_mask]
        iona_min_dist = self.iona_get_min_dist(iona_data, iona_idx_mask)
        iona_noise_std = self.iona_get_noise_std(iona_min_dist)
        iona_flip_prob = self.iona_get_flip_prob(iona_min_dist)
        iona_chi_native = torch.where(
            torch.randn_like(iona_data["chi"]) > 0,
            iona_data["chi"],
            iona_data["chi_alt"],
        )
        iona_chi = iona_chi_native.clone()
        iona_chi_mask = iona_data["chi_mask"]
        if self.iona_add_noise:
            iona_chi = self.iona_add_chi_gaussian_noise(
                iona_chi, iona_noise_std, iona_chi_mask
            )
            iona_chi = self.iona_random_flip_chi(
                iona_chi, iona_flip_prob, iona_chi_mask
            )
        iona_chi[iona_idx_mask] = 0.0
        iona_corrupt_flag = torch.zeros(iona_L, dtype=torch.bool)
        iona_corrupt_flag[iona_idx_mask] = True
        iona_corrupt_flag[iona_min_dist <= 8] = True
        iona_masked_flag = torch.zeros(iona_L, dtype=torch.bool)
        iona_masked_flag[iona_idx_mask] = True
        iona_data["chi_native"] = iona_chi_native
        iona_data["chi_corrupt"] = iona_chi
        iona_data["chi_corrupt_flag"] = iona_corrupt_flag
        iona_data["chi_masked_flag"] = iona_masked_flag
        return iona_data
