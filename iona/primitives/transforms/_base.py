import copy
import torch


class IONATransformSequence:

    def __init__(self, iona_transforms):
        self.iona_transforms = iona_transforms

    def __call__(self, iona_data):
        for iona_t in self.iona_transforms:
            iona_data = iona_t(iona_data)
        return iona_data


iona_TRANSFORM_DICT = {}


def iona_register_transform(iona_name):

    def iona_decorator(cls):
        iona_TRANSFORM_DICT[iona_name] = cls
        return cls

    return iona_decorator


def iona_build_transforms(iona_cfg):
    if iona_cfg is None or len(iona_cfg) == 0:
        return None
    iona_tfms = []
    for iona_t_dict in iona_cfg:
        iona_t_dict = copy.deepcopy(iona_t_dict)
        cls = iona_TRANSFORM_DICT[iona_t_dict.pop("type")]
        iona_tfms.append(cls(**iona_t_dict))
    return IONATransformSequence(iona_tfms)


def iona_index_select(iona_v, iona_index, iona_n):
    if isinstance(iona_v, torch.Tensor) and iona_v.size(0) == iona_n:
        return iona_v[iona_index]
    elif isinstance(iona_v, list) and len(iona_v) == iona_n:
        return [iona_v[iona_i] for iona_i in iona_index]
    else:
        return iona_v


def iona_index_select_data(iona_data, iona_index):
    return {
        iona_k: iona_index_select(iona_v, iona_index, iona_data["aa"].size(0))
        for iona_k, iona_v in iona_data.items()
    }


def iona_mask_select(iona_v, iona_mask):
    if isinstance(iona_v, torch.Tensor) and iona_v.size(0) == iona_mask.size(0):
        return iona_v[iona_mask]
    elif isinstance(iona_v, list) and len(iona_v) == iona_mask.size(0):
        return [iona_v[iona_i] for iona_i, iona_b in enumerate(iona_mask) if iona_b]
    else:
        return iona_v


def iona_mask_select_data(iona_data, iona_mask):
    return {
        iona_k: iona_mask_select(iona_v, iona_mask)
        for iona_k, iona_v in iona_data.items()
    }


def iona_get_CB_positions(iona_pos_atoms, iona_mask_atoms):
    from iona.primitives.protein.constants import IONABackboneAtom

    iona_L = iona_pos_atoms.size(0)
    iona_pos_CA = iona_pos_atoms[:, IONABackboneAtom.CA]
    if iona_pos_atoms.size(1) < 5:
        return iona_pos_CA
    iona_pos_CB = iona_pos_atoms[:, IONABackboneAtom.CB]
    iona_mask_CB = iona_mask_atoms[:, IONABackboneAtom.CB, None].expand(iona_L, 3)
    return torch.where(iona_mask_CB, iona_pos_CB, iona_pos_CA)
