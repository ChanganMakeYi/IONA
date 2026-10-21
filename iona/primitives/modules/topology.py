import torch
import torch.nn.functional as F


def iona_get_consecutive_flag(iona_chain_nb, iona_res_nb, iona_mask):
    iona_d_res_nb = (iona_res_nb[:, 1:] - iona_res_nb[:, :-1]).abs()
    iona_same_chain = iona_chain_nb[:, 1:] == iona_chain_nb[:, :-1]
    iona_consec = torch.logical_and(iona_d_res_nb == 1, iona_same_chain)
    iona_consec = torch.logical_and(iona_consec, iona_mask[:, :-1])
    return iona_consec


def iona_get_terminus_flag(iona_chain_nb, iona_res_nb, iona_mask):
    iona_consec = iona_get_consecutive_flag(iona_chain_nb, iona_res_nb, iona_mask)
    iona_N_term_flag = F.pad(torch.logical_not(iona_consec), pad=(1, 0), value=1)
    iona_C_term_flag = F.pad(torch.logical_not(iona_consec), pad=(0, 1), value=1)
    return (iona_N_term_flag, iona_C_term_flag)
