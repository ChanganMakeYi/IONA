import random
import torch
from ._base import (
    iona_index_select_data,
    iona_register_transform,
    iona_get_CB_positions,
)


@iona_register_transform("focused_random_patch")
class IONAFocusedRandomPatch(object):

    def __init__(self, iona_focus_attr, iona_seed_nbh_size=32, iona_patch_size=128):
        super().__init__()
        self.iona_focus_attr = iona_focus_attr
        self.iona_seed_nbh_size = iona_seed_nbh_size
        self.iona_patch_size = iona_patch_size

    def __call__(self, iona_data):
        iona_focus_flag = iona_data[self.iona_focus_attr] > 0
        if iona_focus_flag.sum() == 0:
            iona_focus_flag[random.randint(0, iona_focus_flag.size(0) - 1)] = True
        iona_seed_idx = torch.multinomial(iona_focus_flag.float(), num_samples=1).item()
        iona_pos_CB = iona_get_CB_positions(
            iona_data["pos_atoms"], iona_data["mask_atoms"]
        )
        iona_pos_seed = iona_pos_CB[iona_seed_idx : iona_seed_idx + 1]
        iona_dist_from_seed = torch.cdist(iona_pos_CB, iona_pos_seed)[:, 0]
        iona_nbh_seed_idx = iona_dist_from_seed.argsort()[: self.iona_seed_nbh_size]
        iona_core_idx = iona_nbh_seed_idx[iona_focus_flag[iona_nbh_seed_idx]]
        iona_dist_from_core = torch.cdist(iona_pos_CB, iona_pos_CB[iona_core_idx]).min(
            dim=1
        )[0]
        iona_patch_idx = iona_dist_from_core.argsort()[: self.iona_patch_size]
        iona_patch_idx = iona_patch_idx.sort()[0]
        iona_core_flag = torch.zeros([iona_data["aa"].size(0)], dtype=torch.bool)
        iona_core_flag[iona_core_idx] = True
        iona_data["core_flag"] = iona_core_flag
        iona_data_patch = iona_index_select_data(iona_data, iona_patch_idx)
        return iona_data_patch


@iona_register_transform("random_patch")
class IONARandomPatch(object):

    def __init__(self, iona_seed_nbh_size=32, iona_patch_size=128):
        super().__init__()
        self.iona_seed_nbh_size = iona_seed_nbh_size
        self.iona_patch_size = iona_patch_size

    def __call__(self, iona_data):
        iona_seed_idx = random.randint(0, iona_data["aa"].size(0) - 1)
        iona_pos_CB = iona_get_CB_positions(
            iona_data["pos_atoms"], iona_data["mask_atoms"]
        )
        iona_pos_seed = iona_pos_CB[iona_seed_idx : iona_seed_idx + 1]
        iona_dist_from_seed = torch.cdist(iona_pos_CB, iona_pos_seed)[:, 0]
        iona_core_idx = iona_dist_from_seed.argsort()[: self.iona_seed_nbh_size]
        iona_dist_from_core = torch.cdist(iona_pos_CB, iona_pos_CB[iona_core_idx]).min(
            dim=1
        )[0]
        iona_patch_idx = iona_dist_from_core.argsort()[: self.iona_patch_size]
        iona_patch_idx = iona_patch_idx.sort()[0]
        iona_core_flag = torch.zeros([iona_data["aa"].size(0)], dtype=torch.bool)
        iona_core_flag[iona_core_idx] = True
        iona_data["core_flag"] = iona_core_flag
        iona_data_patch = iona_index_select_data(iona_data, iona_patch_idx)
        return iona_data_patch


@iona_register_transform("selected_region_with_padding_patch")
class IONASelectedRegionWithPaddingPatch(object):

    def __init__(
        self, iona_select_attr, iona_each_residue_nbh_size, iona_patch_size_limit
    ):
        super().__init__()
        self.iona_select_attr = iona_select_attr
        self.iona_each_residue_nbh_size = iona_each_residue_nbh_size
        self.iona_patch_size_limit = iona_patch_size_limit

    def __call__(self, iona_data):
        iona_select_flag = iona_data[self.iona_select_attr] > 0
        iona_pos_CB = iona_get_CB_positions(
            iona_data["pos_atoms"], iona_data["mask_atoms"]
        )
        iona_pos_sel = iona_pos_CB[iona_select_flag]
        iona_dist_from_sel = torch.cdist(iona_pos_CB, iona_pos_sel)
        iona_nbh_sel_idx = torch.argsort(iona_dist_from_sel, dim=0)[
            : self.iona_each_residue_nbh_size, :
        ]
        iona_patch_idx = iona_nbh_sel_idx.view(-1).unique()
        iona_data_patch = iona_index_select_data(iona_data, iona_patch_idx)
        return iona_data_patch


@iona_register_transform("selected_region_fixed_size_patch")
class IONASelectedRegionFixedSizePatch(object):

    def __init__(self, iona_select_attr, iona_patch_size):
        super().__init__()
        self.iona_select_attr = iona_select_attr
        self.iona_patch_size = iona_patch_size

    def __call__(self, iona_data):
        iona_select_flag = iona_data[self.iona_select_attr] > 0
        if iona_select_flag.sum() == 0:
            iona_select_flag = iona_select_flag.clone()
            iona_select_flag[0] = True
        iona_pos_CB = iona_get_CB_positions(
            iona_data["pos_atoms"], iona_data["mask_atoms"]
        )
        iona_pos_sel = iona_pos_CB[iona_select_flag]
        iona_dist_from_sel = torch.cdist(iona_pos_CB, iona_pos_sel).min(dim=1)[0]
        iona_patch_idx = torch.argsort(iona_dist_from_sel)[: self.iona_patch_size]
        iona_data_patch = iona_index_select_data(iona_data, iona_patch_idx)
        return iona_data_patch
