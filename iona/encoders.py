import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from iona.primitives.protein.constants import IONABackboneAtom
from iona.primitives.modules.layers import IONALayerNorm, IONAAngularEncoding
from iona.primitives.modules.geometry import (
    iona_global_to_local,
    iona_local_to_global,
    iona_construct_3d_basis,
    iona_angstrom_to_nm,
    iona_pairwise_dihedrals,
    iona_per_directions,
    iona_pairwise_directions,
)

iona_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IONAResidueEncoder(nn.Module):

    def __init__(self, iona_feat_dim, iona_args, iona_max_aa_types=22):
        super().__init__()
        self.iona_args = iona_args
        self.iona_aatype_embed = nn.Embedding(iona_max_aa_types, iona_feat_dim)
        self.iona_dihed_embed = IONAAngularEncoding()
        self.iona_mut_embed = nn.Embedding(
            num_embeddings=2, embedding_dim=int(iona_feat_dim / 2), padding_idx=0
        )
        iona_infeat_dim = (
            iona_feat_dim
            + self.iona_dihed_embed.iona_get_out_dim(7)
            + 3 * 3
            + int(iona_feat_dim / 2)
        )
        self.iona_out_mlp = nn.Sequential(
            nn.Linear(iona_infeat_dim, iona_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_feat_dim, iona_feat_dim),
        )

    def forward(self, iona_batch, iona_pretrain=False):
        iona_N, iona_L = iona_batch["aa"].size()
        iona_mask_residue = iona_batch["mask_atoms"][:, :, IONABackboneAtom.CA]
        if iona_pretrain:
            iona_input_aa = copy.deepcopy(iona_batch["aa"])
            iona_mask_aa = (
                torch.bernoulli(
                    torch.full(
                        size=(iona_N, iona_L), fill_value=self.iona_args.iona_mask_ratio
                    )
                ).to(iona_device)
                * iona_batch["mask_atoms"][:, :, 1]
            ).bool()
            iona_aa_noise = torch.randint(0, 22, size=(iona_N, iona_L)).to(iona_device)
            iona_input_aa[iona_mask_aa] = iona_aa_noise[iona_mask_aa]
        else:
            iona_input_aa = iona_batch["aa"]
        iona_aa_feat = self.iona_aatype_embed(iona_input_aa)
        iona_dihedral = torch.cat(
            [
                iona_batch["phi"][..., None],
                iona_batch["psi"][..., None],
                iona_batch["omega"][..., None],
                iona_batch["chi"],
            ],
            dim=-1,
        )
        iona_dihedral_mask = torch.cat(
            [
                iona_batch["phi_mask"][..., None],
                iona_batch["psi_mask"][..., None],
                iona_batch["omega_mask"][..., None],
                iona_batch["chi_mask"],
            ],
            dim=-1,
        )
        iona_dihedral_feat = (
            self.iona_dihed_embed(iona_dihedral[..., None])
            * iona_dihedral_mask[..., None]
        )
        iona_dihedral_feat = iona_dihedral_feat.reshape(iona_N, iona_L, -1)
        iona_direct_feat = iona_per_directions(iona_batch["pos_atoms"])
        if iona_pretrain:
            iona_input_dihedral_feat = copy.deepcopy(iona_dihedral_feat)
            iona_mask_angle = (
                torch.bernoulli(
                    torch.full(
                        size=(iona_N, iona_L), fill_value=self.iona_args.iona_mask_ratio
                    )
                ).to(iona_device)
                * iona_batch["mask_atoms"][:, :, 1]
            ).bool()
            iona_input_dihedral_feat[iona_mask_angle] *= 0
        else:
            iona_input_dihedral_feat = iona_dihedral_feat
        iona_mut_feat = self.iona_mut_embed(iona_batch["mut_flag"].long())
        iona_out_feat = self.iona_out_mlp(
            torch.cat(
                [
                    iona_aa_feat,
                    iona_input_dihedral_feat,
                    iona_direct_feat,
                    iona_mut_feat,
                ],
                dim=-1,
            )
        )
        iona_out_feat = iona_out_feat * iona_mask_residue[:, :, None]
        if iona_pretrain:
            return (iona_out_feat, [iona_dihedral_feat, iona_mask_aa, iona_mask_angle])
        else:
            return iona_out_feat


class IONAPairEncoder(nn.Module):

    def __init__(
        self,
        iona_feat_dim,
        iona_max_num_atoms=5,
        iona_max_aa_types=22,
        iona_max_relpos=32,
    ):
        super().__init__()
        self.iona_max_num_atoms = iona_max_num_atoms
        self.iona_max_aa_types = iona_max_aa_types
        self.iona_max_relpos = iona_max_relpos
        self.iona_aa_pair_embed = nn.Embedding(
            self.iona_max_aa_types * self.iona_max_aa_types, iona_feat_dim
        )
        self.iona_relpos_embed = nn.Embedding(2 * iona_max_relpos + 1, iona_feat_dim)
        self.iona_aapair_to_distcoef = nn.Embedding(
            self.iona_max_aa_types * self.iona_max_aa_types,
            iona_max_num_atoms * iona_max_num_atoms,
        )
        nn.init.zeros_(self.iona_aapair_to_distcoef.weight)
        self.iona_distance_embed = nn.Sequential(
            nn.Linear(iona_max_num_atoms * iona_max_num_atoms, iona_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_feat_dim, iona_feat_dim),
            nn.ReLU(),
        )
        self.iona_dihedral_embed = IONAAngularEncoding()
        iona_feat_dihed_dim = self.iona_dihedral_embed.iona_get_out_dim(3)
        iona_infeat_dim = (
            iona_feat_dim + iona_feat_dim + iona_feat_dim + iona_feat_dihed_dim + 3 * 4
        )
        self.iona_out_mlp = nn.Sequential(
            nn.Linear(iona_infeat_dim, iona_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_feat_dim, iona_feat_dim),
        )

    def forward(self, iona_batch):
        iona_N, iona_L = iona_batch["aa"].size()
        iona_mask_residue = iona_batch["mask_atoms"][:, :, IONABackboneAtom.CA]
        iona_mask_pair = iona_mask_residue[:, :, None] * iona_mask_residue[:, None, :]
        iona_aa_pair = (
            iona_batch["aa"][:, :, None] * self.iona_max_aa_types
            + iona_batch["aa"][:, None, :]
        )
        iona_aa_pair_feat = self.iona_aa_pair_embed(iona_aa_pair)
        iona_same_chain = (
            iona_batch["chain_nb"][:, :, None] == iona_batch["chain_nb"][:, None, :]
        )
        iona_relpos = torch.clamp(
            iona_batch["res_nb"][:, :, None] - iona_batch["res_nb"][:, None, :],
            min=-self.iona_max_relpos,
            max=self.iona_max_relpos,
        )
        iona_relpos_feat = (
            self.iona_relpos_embed(iona_relpos + self.iona_max_relpos)
            * iona_same_chain[:, :, :, None]
        )
        iona_d = iona_angstrom_to_nm(
            torch.linalg.norm(
                iona_batch["pos_atoms"][:, :, None, :, None]
                - iona_batch["pos_atoms"][:, None, :, None, :],
                dim=-1,
                ord=2,
            )
        ).reshape(iona_N, iona_L, iona_L, -1)
        iona_c = F.softplus(self.iona_aapair_to_distcoef(iona_aa_pair))
        iona_d_gauss = torch.exp(-1 * iona_c * iona_d**2)
        iona_mask_atom_pair = (
            iona_batch["mask_atoms"][:, :, None, :, None]
            * iona_batch["mask_atoms"][:, None, :, None, :]
        ).reshape(iona_N, iona_L, iona_L, -1)
        iona_dist_feat = self.iona_distance_embed(iona_d_gauss * iona_mask_atom_pair)
        iona_dihed = iona_pairwise_dihedrals(iona_batch["pos_atoms"])
        iona_dihed_feat = self.iona_dihedral_embed(iona_dihed)
        iona_direct_feat = iona_pairwise_directions(iona_batch["pos_atoms"])
        iona_feat_all = self.iona_out_mlp(
            torch.cat(
                [
                    iona_aa_pair_feat,
                    iona_relpos_feat,
                    iona_dist_feat,
                    iona_dihed_feat,
                    iona_direct_feat,
                ],
                dim=-1,
            )
        )
        iona_feat_all = iona_feat_all * iona_mask_pair[:, :, :, None]
        return iona_feat_all


def iona_masked_attention(iona_logits, iona_mask, iona_inf=100000.0):
    iona_mask_pair = iona_mask.unsqueeze(-1).expand_as(iona_logits)
    iona_logits = torch.where(iona_mask_pair, iona_logits, iona_logits - iona_inf)
    iona_alpha = torch.softmax(iona_logits, dim=2)
    iona_alpha = torch.where(iona_mask_pair, iona_alpha, torch.zeros_like(iona_alpha))
    return iona_alpha


def iona_split_heads(iona_x, iona_n_heads, iona_n_ch):
    iona_s = list(iona_x.size())[:-1] + [iona_n_heads, iona_n_ch]
    return iona_x.view(*iona_s)


class IONAGeometricAttentionBlock(nn.Module):

    def __init__(
        self,
        iona_node_feat_dim,
        iona_pair_feat_dim,
        iona_codebook_head=1,
        iona_hidden_dim=32,
        iona_num_points=8,
        iona_num_heads=8,
        iona_bias=False,
    ):
        super().__init__()
        self.iona_hidden_dim = iona_hidden_dim
        self.iona_num_points = iona_num_points
        self.iona_num_heads = iona_num_heads
        self.iona_proj_query = nn.Linear(
            iona_node_feat_dim, iona_hidden_dim * iona_num_heads, bias=iona_bias
        )
        self.iona_proj_key = nn.Linear(
            iona_node_feat_dim, iona_hidden_dim * iona_num_heads, bias=iona_bias
        )
        self.iona_proj_value = nn.Linear(
            iona_node_feat_dim, iona_hidden_dim * iona_num_heads, bias=iona_bias
        )
        self.iona_proj_pair = nn.Linear(
            iona_pair_feat_dim, iona_num_heads, bias=iona_bias
        )
        self.iona_spatial_coef = nn.Parameter(
            torch.full([1, 1, 1, iona_num_heads], fill_value=np.log(np.exp(1.0) - 1.0)),
            requires_grad=True,
        )
        self.iona_proj_query_point = nn.Linear(
            iona_node_feat_dim, iona_num_points * iona_num_heads * 3, bias=iona_bias
        )
        self.iona_proj_key_point = nn.Linear(
            iona_node_feat_dim, iona_num_points * iona_num_heads * 3, bias=iona_bias
        )
        self.iona_proj_value_point = nn.Linear(
            iona_node_feat_dim, iona_num_points * iona_num_heads * 3, bias=iona_bias
        )
        self.iona_mlp_transition_1 = nn.Linear(
            iona_num_heads * iona_pair_feat_dim
            + iona_num_heads * iona_hidden_dim
            + iona_num_heads * iona_num_points,
            iona_node_feat_dim,
        )
        self.iona_layer_norm_1 = IONALayerNorm(iona_node_feat_dim)
        self.iona_mlp_transition_2 = nn.Sequential(
            nn.Linear(iona_node_feat_dim, iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_node_feat_dim, iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_node_feat_dim, iona_node_feat_dim * iona_codebook_head),
        )
        self.iona_layer_norm_2 = IONALayerNorm(iona_node_feat_dim * iona_codebook_head)

    def iona_attention_logits(self, iona_R, iona_t, iona_x, iona_z):
        iona_N, iona_L, _ = iona_t.size()
        iona_query_node = iona_split_heads(
            self.iona_proj_query(iona_x), self.iona_num_heads, self.iona_hidden_dim
        )
        iona_key_node = iona_split_heads(
            self.iona_proj_key(iona_x), self.iona_num_heads, self.iona_hidden_dim
        )
        iona_logits_node = (
            iona_query_node.unsqueeze(2)
            * iona_key_node.unsqueeze(1)
            * (1 / np.sqrt(self.iona_hidden_dim))
        ).sum(-1)
        iona_logits_pair = self.iona_proj_pair(iona_z)
        iona_query_point = iona_split_heads(
            self.iona_proj_query_point(iona_x),
            self.iona_num_heads * self.iona_num_points,
            3,
        )
        iona_query_point = iona_local_to_global(iona_R, iona_t, iona_query_point)
        iona_query_point = iona_query_point.reshape(
            iona_N, iona_L, self.iona_num_heads, -1
        )
        iona_key_point = iona_split_heads(
            self.iona_proj_key_point(iona_x),
            self.iona_num_heads * self.iona_num_points,
            3,
        )
        iona_key_point = iona_local_to_global(iona_R, iona_t, iona_key_point)
        iona_key_point = iona_key_point.reshape(iona_N, iona_L, self.iona_num_heads, -1)
        iona_sum_sq_dist = (
            (iona_query_point.unsqueeze(2) - iona_key_point.unsqueeze(1)) ** 2
        ).sum(-1)
        iona_gamma = F.softplus(self.iona_spatial_coef)
        iona_logits_spatial = iona_sum_sq_dist * (
            -1 * iona_gamma * np.sqrt(2 / (9 * self.iona_num_points)) / 2
        )
        return (iona_logits_node + iona_logits_pair + iona_logits_spatial) * np.sqrt(
            1 / 3
        )

    def iona_node_aggregation(self, iona_alpha, iona_R, iona_t, iona_x, iona_z):
        iona_N, iona_L = iona_x.shape[:2]
        iona_value_node = iona_split_heads(
            self.iona_proj_value(iona_x), self.iona_num_heads, self.iona_hidden_dim
        )
        iona_feat_node = iona_alpha.unsqueeze(-1) * iona_value_node.unsqueeze(1)
        iona_feat_node = iona_feat_node.sum(dim=2).reshape(iona_N, iona_L, -1)
        iona_feat_pair = iona_alpha.unsqueeze(-1) * iona_z.unsqueeze(-2)
        iona_feat_pair = iona_feat_pair.sum(dim=2).reshape(iona_N, iona_L, -1)
        iona_value_point = iona_split_heads(
            self.iona_proj_value_point(iona_x),
            self.iona_num_heads * self.iona_num_points,
            3,
        )
        iona_value_point = iona_local_to_global(iona_R, iona_t, iona_value_point)
        iona_value_point = iona_value_point.reshape(
            iona_N, iona_L, self.iona_num_heads, self.iona_num_points, -1
        )
        iona_feat_spatial = iona_alpha.unsqueeze(-1).unsqueeze(
            -1
        ) * iona_value_point.unsqueeze(1)
        iona_feat_spatial = iona_feat_spatial.sum(dim=2)
        iona_feat_spatial = iona_global_to_local(iona_R, iona_t, iona_feat_spatial)
        iona_feat_spatial = iona_feat_spatial.norm(dim=-1).reshape(iona_N, iona_L, -1)
        return torch.cat([iona_feat_node, iona_feat_pair, iona_feat_spatial], dim=-1)

    def forward(self, iona_R, iona_t, iona_x, iona_z, iona_intra_mask, iona_inter_mask):
        iona_att_logits = self.iona_attention_logits(iona_R, iona_t, iona_x, iona_z)
        iona_att_ratio = (
            (
                iona_intra_mask.sum(-1).sum(-1)
                / (iona_intra_mask.sum(-1).sum(-1) + iona_inter_mask.sum(-1).sum(-1))
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        iona_alpha = iona_masked_attention(
            iona_att_logits, iona_intra_mask
        ) * iona_att_ratio + iona_masked_attention(iona_att_logits, iona_inter_mask) * (
            1 - iona_att_ratio
        )
        iona_agg_feats = self.iona_node_aggregation(
            iona_alpha, iona_R, iona_t, iona_x, iona_z
        )
        iona_feats = self.iona_layer_norm_1(
            iona_x + self.iona_mlp_transition_1(iona_agg_feats)
        )
        iona_out_feats = self.iona_mlp_transition_2(iona_feats)
        if iona_feats.shape[-1] == iona_out_feats.shape[-1]:
            iona_feats = self.iona_layer_norm_2(iona_feats + iona_out_feats)
        else:
            iona_feats = self.iona_layer_norm_2(iona_out_feats)
        return iona_feats


class IONAGeometricEncoder(nn.Module):

    def __init__(
        self,
        iona_node_feat_dim,
        iona_pair_feat_dim,
        iona_num_layers,
        iona_args,
        iona_codebook_head=1,
    ):
        super(IONAGeometricEncoder, self).__init__()
        self.iona_blocks = nn.ModuleList()
        for iona_i in range(iona_num_layers):
            if iona_i == iona_num_layers - 1:
                self.iona_blocks.append(
                    IONAGeometricAttentionBlock(
                        iona_node_feat_dim, iona_pair_feat_dim, iona_codebook_head
                    )
                )
            else:
                self.iona_blocks.append(
                    IONAGeometricAttentionBlock(iona_node_feat_dim, iona_pair_feat_dim)
                )
        self.iona_args = iona_args

    def iona_build_neighborhood(self, iona_batch, iona_cutoff=15.0, iona_K=15):
        iona_mask_residue = iona_batch["mask_atoms"][:, :, IONABackboneAtom.CA]
        iona_mask_pair = iona_mask_residue[:, :, None] * iona_mask_residue[:, None, :]
        iona_same_chain = (
            iona_batch["chain_nb"][:, :, None] == iona_batch["chain_nb"][:, None, :]
        )
        iona_diff_chain = (
            iona_batch["chain_nb"][:, :, None] != iona_batch["chain_nb"][:, None, :]
        )
        iona_seq_dis = torch.abs(
            iona_batch["res_nb"][:, :, None] - iona_batch["res_nb"][:, None, :]
        )
        iona_seq_mask = (iona_seq_dis == torch.tensor(1)) + (
            iona_seq_dis == torch.tensor(2)
        )
        iona_pairwise_dis = torch.linalg.norm(
            iona_batch["pos_atoms"][:, :, None, IONABackboneAtom.CA]
            - iona_batch["pos_atoms"][:, None, :, IONABackboneAtom.CA],
            dim=-1,
            ord=2,
        )
        iona_rball_mask = iona_pairwise_dis < iona_cutoff
        iona_dis, _ = iona_pairwise_dis.topk(iona_K + 1, dim=-1, largest=False)
        iona_knn_mask = iona_pairwise_dis < iona_dis[:, :, iona_K : iona_K + 1]
        iona_intra_mask = (
            (iona_rball_mask + iona_knn_mask + iona_seq_mask)
            * iona_same_chain
            * iona_mask_pair
        ).bool()
        iona_inter_mask = (
            (iona_rball_mask + iona_knn_mask) * iona_diff_chain * iona_mask_pair
        ).bool()
        return (iona_intra_mask, iona_inter_mask)

    def forward(self, iona_batch, iona_res_feat, iona_pair_feat):
        iona_pos_atoms = iona_batch["pos_atoms"]
        iona_R = iona_construct_3d_basis(
            iona_pos_atoms[:, :, IONABackboneAtom.CA],
            iona_pos_atoms[:, :, IONABackboneAtom.C],
            iona_pos_atoms[:, :, IONABackboneAtom.N],
        )
        iona_t = iona_angstrom_to_nm(iona_pos_atoms[:, :, IONABackboneAtom.CA])
        iona_intra_mask, iona_inter_mask = self.iona_build_neighborhood(
            iona_batch,
            iona_cutoff=self.iona_args.iona_cutoff,
            iona_K=self.iona_args.iona_knn,
        )
        for iona_block in self.iona_blocks:
            iona_res_feat = iona_block(
                iona_R,
                iona_t,
                iona_res_feat,
                iona_pair_feat,
                iona_intra_mask,
                iona_inter_mask,
            )
        return iona_res_feat


class IONASpatialUpdateBlock(nn.Module):

    def __init__(
        self,
        iona_node_feat_dim,
        iona_pair_feat_dim,
        iona_n_channel=4,
        iona_edge_type=2,
        iona_dropout=0.1,
    ):
        super().__init__()
        self.iona_dropout = nn.Dropout(iona_dropout)
        self.iona_coord_mlp = nn.ModuleList()
        self.iona_relation_mlp = nn.ModuleList()
        self.iona_message_mlp = nn.Sequential(
            nn.Linear(
                iona_node_feat_dim * 2 + iona_n_channel**2 + iona_pair_feat_dim,
                iona_node_feat_dim,
            ),
            nn.SiLU(),
            nn.Linear(iona_node_feat_dim, iona_node_feat_dim),
        )
        self.iona_node_mlp = nn.Sequential(
            nn.Linear(iona_node_feat_dim * 3, iona_node_feat_dim),
            nn.SiLU(),
            nn.Linear(iona_node_feat_dim, iona_node_feat_dim),
        )
        self.iona_edge_mlp = nn.Sequential(
            nn.Linear(
                iona_node_feat_dim + iona_pair_feat_dim + iona_node_feat_dim,
                iona_node_feat_dim,
            ),
            nn.SiLU(),
            nn.Linear(iona_node_feat_dim, iona_pair_feat_dim),
        )
        for _ in range(iona_edge_type):
            self.iona_relation_mlp.append(
                nn.Linear(iona_node_feat_dim, iona_node_feat_dim, bias=False)
            )
            iona_layer = nn.Linear(iona_node_feat_dim, iona_n_channel, bias=False)
            torch.nn.init.xavier_uniform_(iona_layer.weight, gain=0.001)
            self.iona_coord_mlp.append(
                nn.Sequential(
                    nn.Linear(iona_node_feat_dim, iona_node_feat_dim),
                    nn.SiLU(),
                    iona_layer,
                )
            )
        self.iona_layer_norm = IONALayerNorm(iona_node_feat_dim)

    def iona_coord2radial(self, iona_coord):
        iona_coord_diff = iona_coord[:, :, None] - iona_coord[:, None, :]
        iona_radial = torch.einsum(
            "ijkmn,ijknp->ijkmp", iona_coord_diff, iona_coord_diff.transpose(-1, -2)
        )
        iona_radial = F.normalize(
            iona_radial.reshape(-1, iona_coord.shape[-2], iona_coord.shape[-2]), dim=0
        ).reshape(iona_radial.shape)
        return (iona_radial, iona_coord_diff)

    def iona_message_updating(self, iona_h, iona_z, iona_radial):
        iona_N, iona_L, _ = iona_h.size()
        iona_radial = iona_radial.reshape(iona_N, iona_L, iona_L, -1)
        iona_out = torch.cat(
            [
                iona_h[:, :, None].expand(-1, -1, iona_L, -1),
                iona_h[:, None, :].expand(-1, iona_L, -1, -1),
                iona_radial,
                iona_z,
            ],
            dim=-1,
        )
        iona_out = self.iona_message_mlp(iona_out)
        iona_out = self.iona_dropout(iona_out)
        return iona_out

    def iona_node_updating(self, iona_h, iona_m, iona_intra_mask, iona_inter_mask):
        iona_agg_intra = self.iona_relation_mlp[0](
            (iona_m * iona_intra_mask[:, :, :, None]).sum(dim=-2)
        )
        iona_agg_inter = self.iona_relation_mlp[1](
            (iona_m * iona_inter_mask[:, :, :, None]).sum(dim=-2)
        )
        iona_agg = torch.cat([iona_h, iona_agg_intra, iona_agg_inter], dim=-1)
        iona_out = self.iona_node_mlp(iona_agg)
        iona_out = self.iona_dropout(iona_out)
        iona_out = self.iona_layer_norm(iona_h + iona_out)
        return iona_out

    def iona_edge_updating(self, iona_h, iona_z):
        _, iona_L, _ = iona_h.size()
        iona_out = torch.cat(
            [
                iona_h[:, :, None].expand(-1, -1, iona_L, -1),
                iona_h[:, None, :].expand(-1, iona_L, -1, -1),
                iona_z,
            ],
            dim=-1,
        )
        iona_out = self.iona_edge_mlp(iona_out)
        return iona_out

    def iona_coord_updating(
        self,
        iona_coord,
        iona_coord_diff,
        iona_m,
        iona_intra_mask,
        iona_inter_mask,
        iona_R,
        iona_t,
    ):
        iona_agg_intra = iona_coord_diff * self.iona_coord_mlp[0](iona_m).unsqueeze(-1)
        iona_agg_inter = iona_coord_diff * self.iona_coord_mlp[1](iona_m).unsqueeze(-1)
        iona_agg = (
            iona_agg_intra * iona_intra_mask.unsqueeze(-1).unsqueeze(-1)
            + iona_agg_inter * iona_inter_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=2)
        iona_agg = iona_agg / (iona_intra_mask + iona_inter_mask + 1e-06).sum(
            dim=2
        ).unsqueeze(-1).unsqueeze(-1)
        iona_coord = iona_coord + iona_agg
        return iona_coord

    def forward(
        self,
        iona_R,
        iona_t,
        iona_coord,
        iona_h,
        iona_z,
        iona_intra_mask,
        iona_inter_mask,
    ):
        iona_radial, iona_coord_diff = self.iona_coord2radial(iona_coord)
        iona_m = self.iona_message_updating(iona_h, iona_z, iona_radial)
        iona_h = self.iona_node_updating(
            iona_h, iona_m, iona_intra_mask, iona_inter_mask
        )
        iona_z = self.iona_edge_updating(iona_h, iona_z)
        iona_coord = self.iona_coord_updating(
            iona_coord,
            iona_coord_diff,
            iona_m,
            iona_intra_mask,
            iona_inter_mask,
            iona_R,
            iona_t,
        )
        return (iona_coord, iona_h, iona_z)


class IONAGeometricDecoder(nn.Module):

    def __init__(
        self, iona_node_feat_dim, iona_pair_feat_dim, iona_num_layers, iona_args
    ):
        super(IONAGeometricDecoder, self).__init__()
        self.iona_blocks = nn.ModuleList(
            [
                IONASpatialUpdateBlock(iona_node_feat_dim, iona_pair_feat_dim)
                for _ in range(iona_num_layers)
            ]
        )
        self.iona_args = iona_args
        self.iona_noise_sigma = 1.0

    def iona_build_neighborhood(self, iona_batch, iona_cutoff=15.0, iona_K=15):
        iona_mask_residue = iona_batch["mask_atoms"][:, :, IONABackboneAtom.CA]
        iona_mask_pair = iona_mask_residue[:, :, None] * iona_mask_residue[:, None, :]
        iona_same_chain = (
            iona_batch["chain_nb"][:, :, None] == iona_batch["chain_nb"][:, None, :]
        )
        iona_diff_chain = (
            iona_batch["chain_nb"][:, :, None] != iona_batch["chain_nb"][:, None, :]
        )
        iona_seq_dis = torch.abs(
            iona_batch["res_nb"][:, :, None] - iona_batch["res_nb"][:, None, :]
        )
        iona_seq_mask = (iona_seq_dis == torch.tensor(1)) + (
            iona_seq_dis == torch.tensor(2)
        )
        iona_pairwise_dis = torch.linalg.norm(
            iona_batch["pos_atoms"][:, :, None, IONABackboneAtom.CA]
            - iona_batch["pos_atoms"][:, None, :, IONABackboneAtom.CA],
            dim=-1,
            ord=2,
        )
        iona_rball_mask = iona_pairwise_dis < iona_cutoff
        iona_dis, _ = iona_pairwise_dis.topk(iona_K + 1, dim=-1, largest=False)
        iona_knn_mask = iona_pairwise_dis < iona_dis[:, :, iona_K : iona_K + 1]
        iona_intra_mask = (
            (iona_rball_mask + iona_knn_mask + iona_seq_mask)
            * iona_same_chain
            * iona_mask_pair
        ).bool()
        iona_inter_mask = (
            (iona_rball_mask + iona_knn_mask) * iona_diff_chain * iona_mask_pair
        ).bool()
        return (iona_intra_mask, iona_inter_mask)

    def forward(self, iona_batch, iona_res_feat, iona_pair_feat):
        iona_intra_mask, iona_inter_mask = self.iona_build_neighborhood(
            iona_batch,
            iona_cutoff=self.iona_args.iona_cutoff,
            iona_K=self.iona_args.iona_knn,
        )
        iona_coord = copy.deepcopy(iona_batch["pos_atoms"][:, :, :4])
        iona_mask = (
            torch.bernoulli(
                torch.full(
                    size=(iona_coord.shape[0], iona_coord.shape[1]),
                    fill_value=self.iona_args.iona_mask_ratio,
                )
            ).to(iona_device)
            * iona_batch["mask_atoms"][:, :, 1]
        ).bool()
        iona_coord_noise = (
            torch.normal(
                0,
                self.iona_noise_sigma,
                size=(iona_coord.shape[0], iona_coord.shape[1], 1, iona_coord.shape[3]),
            )
            .expand_as(iona_coord)
            .to(iona_device)
        )
        iona_coord[iona_mask] += iona_coord_noise[iona_mask]
        iona_pos_atoms = iona_batch["pos_atoms"]
        iona_R = iona_construct_3d_basis(
            iona_pos_atoms[:, :, IONABackboneAtom.CA],
            iona_pos_atoms[:, :, IONABackboneAtom.C],
            iona_pos_atoms[:, :, IONABackboneAtom.N],
        )
        iona_t = iona_angstrom_to_nm(iona_pos_atoms[:, :, IONABackboneAtom.CA])
        for iona_block in self.iona_blocks:
            iona_coord, iona_res_feat, iona_pair_feat = iona_block(
                iona_R,
                iona_t,
                iona_coord,
                iona_res_feat,
                iona_pair_feat,
                iona_intra_mask,
                iona_inter_mask,
            )
        return (iona_coord, iona_mask)
