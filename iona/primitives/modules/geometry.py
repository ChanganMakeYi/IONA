import torch
import torch.nn.functional as F
from iona.primitives.protein.constants import IONABackboneAtom
from .topology import iona_get_terminus_flag


def iona_safe_norm(
    iona_x, iona_dim=-1, iona_keepdim=False, iona_eps=1e-08, iona_sqrt=True
):
    iona_out = torch.clamp(
        torch.sum(torch.square(iona_x), dim=iona_dim, keepdim=iona_keepdim),
        min=iona_eps,
    )
    return torch.sqrt(iona_out) if iona_sqrt else iona_out


def iona_pairwise_distances(iona_x, iona_y=None, iona_return_v=False):
    if iona_y is None:
        iona_y = iona_x
    iona_v = iona_x.unsqueeze(2) - iona_y.unsqueeze(1)
    iona_d = iona_safe_norm(iona_v, iona_dim=-1)
    if iona_return_v:
        return (iona_d, iona_v)
    else:
        return iona_d


def iona_normalize_vector(iona_v, iona_dim, iona_eps=1e-06):
    return iona_v / (
        torch.linalg.norm(iona_v, ord=2, dim=iona_dim, keepdim=True) + iona_eps
    )


def iona_project_v2v(iona_v, iona_e, iona_dim):
    return (iona_e * iona_v).sum(dim=iona_dim, keepdim=True) * iona_e


def iona_construct_3d_basis(iona_center, iona_p1, iona_p2):
    iona_v1 = iona_p1 - iona_center
    iona_e1 = iona_normalize_vector(iona_v1, iona_dim=-1)
    iona_v2 = iona_p2 - iona_center
    iona_u2 = iona_v2 - iona_project_v2v(iona_v2, iona_e1, iona_dim=-1)
    iona_e2 = iona_normalize_vector(iona_u2, iona_dim=-1)
    iona_e3 = torch.cross(iona_e1, iona_e2, dim=-1)
    iona_mat = torch.cat(
        [iona_e1.unsqueeze(-1), iona_e2.unsqueeze(-1), iona_e3.unsqueeze(-1)], dim=-1
    )
    return iona_mat


def iona_local_to_global(iona_R, iona_t, iona_p):
    assert iona_p.size(-1) == 3
    iona_p_size = iona_p.size()
    iona_N, iona_L = (iona_p_size[0], iona_p_size[1])
    iona_p = iona_p.view(iona_N, iona_L, -1, 3).transpose(-1, -2)
    iona_q = torch.matmul(iona_R, iona_p) + iona_t.unsqueeze(-1)
    iona_q = iona_q.transpose(-1, -2).reshape(iona_p_size)
    return iona_q


def iona_global_to_local(iona_R, iona_t, iona_q):
    assert iona_q.size(-1) == 3
    iona_q_size = iona_q.size()
    iona_N, iona_L = (iona_q_size[0], iona_q_size[1])
    iona_q = iona_q.reshape(iona_N, iona_L, -1, 3).transpose(-1, -2)
    iona_t = iona_t.reshape(iona_N, iona_L, -1, 3).transpose(-1, -2)
    iona_p = torch.matmul(iona_R.transpose(-1, -2), iona_q - iona_t)
    iona_p = iona_p.transpose(-1, -2).reshape(iona_q_size)
    return iona_p


def iona_dihedral_from_four_points(iona_p0, iona_p1, iona_p2, iona_p3):
    iona_v0 = iona_p2 - iona_p1
    iona_v1 = iona_p0 - iona_p1
    iona_v2 = iona_p3 - iona_p2
    iona_u1 = torch.cross(iona_v0, iona_v1, dim=-1)
    iona_n1 = iona_u1 / torch.linalg.norm(iona_u1, dim=-1, keepdim=True)
    iona_u2 = torch.cross(iona_v0, iona_v2, dim=-1)
    iona_n2 = iona_u2 / torch.linalg.norm(iona_u2, dim=-1, keepdim=True)
    iona_sgn = torch.sign((torch.cross(iona_v1, iona_v2, dim=-1) * iona_v0).sum(-1))
    iona_dihed = iona_sgn * torch.acos((iona_n1 * iona_n2).sum(-1))
    iona_dihed = torch.nan_to_num(iona_dihed)
    return iona_dihed


def iona_knn_gather(iona_idx, iona_value):
    iona_N, iona_d = (iona_idx.size(1), iona_value.size(-1))
    iona_idx = iona_idx.unsqueeze(-1).repeat(1, 1, 1, iona_d)
    iona_value = iona_value.unsqueeze(1).repeat(1, iona_N, 1, 1)
    return torch.gather(iona_value, dim=2, index=iona_idx)


def iona_knn_points(iona_q, iona_p, iona_K):
    _, iona_L, _ = iona_p.size()
    iona_d = iona_pairwise_distances(iona_q, iona_p)
    iona_dist, iona_idx = iona_d.topk(min(iona_L, iona_K), dim=-1, largest=False)
    return (iona_dist, iona_idx, iona_knn_gather(iona_idx, iona_p))


def iona_angstrom_to_nm(iona_x):
    return iona_x / 10


def iona_nm_to_angstrom(iona_x):
    return iona_x * 10


def iona_get_backbone_dihedral_angles(
    iona_pos_atoms, iona_chain_nb, iona_res_nb, iona_mask
):
    iona_pos_N = iona_pos_atoms[:, :, IONABackboneAtom.N]
    iona_pos_CA = iona_pos_atoms[:, :, IONABackboneAtom.CA]
    iona_pos_C = iona_pos_atoms[:, :, IONABackboneAtom.C]
    iona_N_term_flag, iona_C_term_flag = iona_get_terminus_flag(
        iona_chain_nb, iona_res_nb, iona_mask
    )
    iona_omega_mask = torch.logical_not(iona_N_term_flag)
    iona_phi_mask = torch.logical_not(iona_N_term_flag)
    iona_psi_mask = torch.logical_not(iona_C_term_flag)
    iona_omega = F.pad(
        iona_dihedral_from_four_points(
            iona_pos_CA[:, :-1],
            iona_pos_C[:, :-1],
            iona_pos_N[:, 1:],
            iona_pos_CA[:, 1:],
        ),
        pad=(1, 0),
        value=0,
    )
    iona_phi = F.pad(
        iona_dihedral_from_four_points(
            iona_pos_C[:, :-1], iona_pos_N[:, 1:], iona_pos_CA[:, 1:], iona_pos_C[:, 1:]
        ),
        pad=(1, 0),
        value=0,
    )
    iona_psi = F.pad(
        iona_dihedral_from_four_points(
            iona_pos_N[:, :-1],
            iona_pos_CA[:, :-1],
            iona_pos_C[:, :-1],
            iona_pos_N[:, 1:],
        ),
        pad=(0, 1),
        value=0,
    )
    iona_mask_bb_dihed = torch.stack(
        [iona_omega_mask, iona_phi_mask, iona_psi_mask], dim=-1
    )
    iona_bb_dihedral = (
        torch.stack([iona_omega, iona_phi, iona_psi], dim=-1) * iona_mask_bb_dihed
    )
    return (iona_bb_dihedral, iona_mask_bb_dihed)


def iona_pairwise_dihedrals(iona_pos_atoms):
    iona_N, iona_L = iona_pos_atoms.shape[:2]
    iona_pos_N = iona_pos_atoms[:, :, IONABackboneAtom.N]
    iona_pos_CA = iona_pos_atoms[:, :, IONABackboneAtom.CA]
    iona_pos_C = iona_pos_atoms[:, :, IONABackboneAtom.C]
    iona_ir_phi = iona_dihedral_from_four_points(
        iona_pos_C[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_N[:, None, :].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_CA[:, None, :].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_C[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_ir_psi = iona_dihedral_from_four_points(
        iona_pos_N[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_CA[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_C[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_N[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_ir_omega = iona_dihedral_from_four_points(
        iona_pos_CA[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_C[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_N[:, None, :].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_CA[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_ir_dihed = torch.stack([iona_ir_phi, iona_ir_psi, iona_ir_omega], dim=-1)
    return iona_ir_dihed


def iona_per_directions(iona_pos_atoms):
    iona_R = iona_construct_3d_basis(
        iona_pos_atoms[:, :, IONABackboneAtom.CA],
        iona_pos_atoms[:, :, IONABackboneAtom.C],
        iona_pos_atoms[:, :, IONABackboneAtom.N],
    )
    iona_pos_N = iona_pos_atoms[:, :, IONABackboneAtom.N]
    iona_pos_CA = iona_pos_atoms[:, :, IONABackboneAtom.CA]
    iona_pos_C = iona_pos_atoms[:, :, IONABackboneAtom.C]
    iona_pos_O = iona_pos_atoms[:, :, IONABackboneAtom.O]
    iona_dire_C = iona_global_to_local(iona_R, iona_pos_CA, iona_pos_C)
    iona_dire_N = iona_global_to_local(iona_R, iona_pos_CA, iona_pos_N)
    iona_dire_O = iona_global_to_local(iona_R, iona_pos_CA, iona_pos_O)
    iona_per_direct_feat = torch.cat([iona_dire_C, iona_dire_N, iona_dire_O], dim=-1)
    return iona_per_direct_feat


def iona_pairwise_directions(iona_pos_atoms):
    iona_R = iona_construct_3d_basis(
        iona_pos_atoms[:, :, IONABackboneAtom.CA],
        iona_pos_atoms[:, :, IONABackboneAtom.C],
        iona_pos_atoms[:, :, IONABackboneAtom.N],
    )
    iona_pos_N = iona_pos_atoms[:, :, IONABackboneAtom.N]
    iona_pos_CA = iona_pos_atoms[:, :, IONABackboneAtom.CA]
    iona_pos_C = iona_pos_atoms[:, :, IONABackboneAtom.C]
    iona_pos_O = iona_pos_atoms[:, :, IONABackboneAtom.O]
    iona_N, iona_L = iona_pos_atoms.shape[:2]
    iona_dire_CA = iona_global_to_local(
        iona_R,
        iona_pos_CA[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_CA[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_dire_C = iona_global_to_local(
        iona_R,
        iona_pos_CA[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_C[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_dire_O = iona_global_to_local(
        iona_R,
        iona_pos_CA[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_O[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_dire_N = iona_global_to_local(
        iona_R,
        iona_pos_CA[:, :, None].expand(iona_N, iona_L, iona_L, 3),
        iona_pos_N[:, None, :].expand(iona_N, iona_L, iona_L, 3),
    )
    iona_pair_direct_feat = torch.cat(
        [iona_dire_CA, iona_dire_C, iona_dire_O, iona_dire_N], dim=-1
    )
    return iona_pair_direct_feat
