import torch
import numpy as np
from ._base import iona_register_transform


@iona_register_transform("add_atom_noise")
class IONAAddAtomNoise(object):

    def __init__(self, iona_noise_std=0.02):
        super().__init__()
        self.iona_noise_std = iona_noise_std

    def __call__(self, iona_data):
        iona_pos_atoms = iona_data["pos_atoms"]
        iona_mask_atoms = iona_data["mask_atoms"]
        iona_noise = (
            torch.randn_like(iona_pos_atoms)
            * self.iona_noise_std
            * iona_mask_atoms[:, :, None]
        )
        iona_pos_noisy = iona_pos_atoms + iona_noise
        iona_data["pos_atoms"] = iona_pos_noisy
        return iona_data


@iona_register_transform("add_chi_angle_noise")
class IONAAddChiAngleNoise(object):

    def __init__(self, iona_noise_std=0.02):
        super().__init__()
        self.iona_noise_std = iona_noise_std

    def iona_normalize_angles(self, iona_angles):
        iona_angles = iona_angles % (2 * np.pi)
        return torch.where(iona_angles > np.pi, iona_angles - 2 * np.pi, iona_angles)

    def __call__(self, iona_data):
        iona_chi, iona_chi_alt = (iona_data["chi"], iona_data["chi_alt"])
        iona_chi_mask = iona_data["chi_mask"]
        iona_get_noise = (
            lambda: torch.randn_like(iona_chi) * self.iona_noise_std * iona_chi_mask
        )
        iona_data["chi"] = self.iona_normalize_angles(iona_chi + iona_get_noise())
        iona_data["chi_alt"] = self.iona_normalize_angles(
            iona_chi_alt + iona_get_noise()
        )
        return iona_data
