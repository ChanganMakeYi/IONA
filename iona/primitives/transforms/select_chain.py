import random
import torch
from ._base import iona_mask_select_data, iona_register_transform


@iona_register_transform("random_interacting_chain")
class IONARandomInteractingChain(object):

    def __init__(self, iona_interaction_attr):
        super().__init__()
        self.iona_interaction_attr = iona_interaction_attr

    def __call__(self, iona_data):
        iona_interact_flag = iona_data[self.iona_interaction_attr] > 0
        if iona_interact_flag.sum() == 0:
            iona_interact_flag[random.randint(0, iona_interact_flag.size(0) - 1)] = True
        iona_seed_idx = torch.multinomial(
            iona_interact_flag.float(), num_samples=1
        ).item()
        iona_chain_nb_selected = iona_data["chain_nb"][iona_seed_idx].item()
        iona_mask_chain = iona_data["chain_nb"] == iona_chain_nb_selected
        return iona_mask_select_data(iona_data, iona_mask_chain)


@iona_register_transform("select_focused")
class IONASelectFocused(object):

    def __init__(self, iona_focus_attr):
        super().__init__()
        self.iona_focus_attr = iona_focus_attr

    def __call__(self, iona_data):
        iona_mask_focus = iona_data[self.iona_focus_attr] > 0
        return iona_mask_select_data(iona_data, iona_mask_focus)
