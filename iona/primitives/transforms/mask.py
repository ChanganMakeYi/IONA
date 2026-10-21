import math
import random
import torch
import torch.nn.functional as F
from ._base import iona_register_transform


def iona_extend_mask(iona_mask, iona_chain_nb):
    iona_mask_sr = torch.logical_and(
        F.pad(iona_mask[:-1], pad=(1, 0), value=0),
        F.pad(iona_chain_nb[:-1], pad=(1, 0), value=-1) == iona_chain_nb,
    )
    iona_mask_sl = torch.logical_and(
        F.pad(iona_mask[1:], pad=(0, 1), value=0),
        F.pad(iona_chain_nb[1:], pad=(0, 1), value=-1) == iona_chain_nb,
    )
    return torch.logical_or(iona_mask, torch.logical_or(iona_mask_sr, iona_mask_sl))


def iona_mask_sidechains(iona_pos_atoms, iona_mask_atoms, iona_mask_idx):
    iona_pos_atoms = iona_pos_atoms.clone()
    iona_pos_atoms[iona_mask_idx, 4:] = 0.0
    iona_mask_atoms = iona_mask_atoms.clone()
    iona_mask_atoms[iona_mask_idx, 4:] = False
    return (iona_pos_atoms, iona_mask_atoms)


@iona_register_transform("random_mask_amino_acids")
class IONARandomMaskAminoAcids(object):

    def __init__(
        self,
        iona_mask_ratio_in_all=0.05,
        iona_ratio_in_maskable_limit=0.5,
        iona_mask_token=20,
        iona_maskable_flag_attr="core_flag",
        iona_extend_maskable_flag=False,
        iona_mask_ratio_mode="constant",
    ):
        super().__init__()
        self.iona_mask_ratio_in_all = iona_mask_ratio_in_all
        self.iona_ratio_in_maskable_limit = iona_ratio_in_maskable_limit
        self.iona_mask_token = iona_mask_token
        self.iona_maskable_flag_attr = iona_maskable_flag_attr
        self.iona_extend_maskable_flag = iona_extend_maskable_flag
        assert iona_mask_ratio_mode in ("constant", "random")
        self.iona_mask_ratio_mode = iona_mask_ratio_mode

    def __call__(self, iona_data):
        if self.iona_maskable_flag_attr is None:
            iona_maskable_flag = torch.ones([iona_data["aa"].size(0)], dtype=torch.bool)
        else:
            iona_maskable_flag = iona_data[self.iona_maskable_flag_attr]
            if self.iona_extend_maskable_flag:
                iona_maskable_flag = iona_extend_mask(
                    iona_maskable_flag, iona_data["chain_nb"]
                )
        iona_num_masked_max = math.ceil(
            self.iona_mask_ratio_in_all * iona_data["aa"].size(0)
        )
        if self.iona_mask_ratio_mode == "random":
            iona_num_masked = random.randint(1, iona_num_masked_max)
        else:
            iona_num_masked = iona_num_masked_max
        iona_mask_idx = torch.multinomial(
            iona_maskable_flag.float() / iona_maskable_flag.sum(),
            num_samples=iona_num_masked,
        )
        iona_mask_idx = iona_mask_idx[
            : math.ceil(
                self.iona_ratio_in_maskable_limit * iona_maskable_flag.sum().item()
            )
        ]
        iona_aa_masked = iona_data["aa"].clone()
        iona_aa_masked[iona_mask_idx] = self.iona_mask_token
        iona_data["aa_true"] = iona_data["aa"]
        iona_data["aa_masked"] = iona_aa_masked
        iona_data["pos_atoms"], iona_data["mask_atoms"] = iona_mask_sidechains(
            iona_data["pos_atoms"], iona_data["mask_atoms"], iona_mask_idx
        )
        return iona_data


@iona_register_transform("mask_selected_amino_acids")
class IONAMaskSelectedAminoAcids(object):

    def __init__(self, iona_select_attr, iona_mask_token=20):
        super().__init__()
        self.iona_select_attr = iona_select_attr
        self.iona_mask_token = iona_mask_token

    def __call__(self, iona_data):
        iona_mask_flag = iona_data[self.iona_select_attr] > 0
        iona_aa_masked = iona_data["aa"].clone()
        iona_aa_masked[iona_mask_flag] = self.iona_mask_token
        iona_data["aa_true"] = iona_data["aa"]
        iona_data["aa_masked"] = iona_aa_masked
        iona_data["pos_atoms"], iona_data["mask_atoms"] = iona_mask_sidechains(
            iona_data["pos_atoms"], iona_data["mask_atoms"], iona_mask_flag
        )
        return iona_data
