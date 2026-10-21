import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

iona_here = os.path.dirname(os.path.abspath(__file__))
for iona_cand in [os.environ.get("IONA_CODE"), os.path.dirname(iona_here)]:
    if iona_cand and os.path.isfile(os.path.join(iona_cand, "encoders.py")):
        sys.path.insert(0, iona_cand)
        break
from iona.encoders import (
    IONAResidueEncoder,
    IONAPairEncoder,
    IONAGeometricAttentionBlock,
    IONAGeometricEncoder,
    iona_split_heads,
    iona_masked_attention,
)
from iona.networks import IONABindingPredictor
from iona.primitives.modules.geometry import (
    iona_local_to_global,
    iona_construct_3d_basis,
    iona_angstrom_to_nm,
)
from iona.primitives.protein.constants import IONABackboneAtom


class IONAScreenedAttentionBlock(IONAGeometricAttentionBlock):

    def __init__(
        self, iona_node_feat_dim, iona_pair_feat_dim, iona_codebook_head=1, **iona_kw
    ):
        super().__init__(
            iona_node_feat_dim, iona_pair_feat_dim, iona_codebook_head, **iona_kw
        )
        self.iona_charge_head = nn.Linear(iona_node_feat_dim, self.iona_num_heads)
        self.iona_log_kappa = nn.Parameter(torch.tensor(0.0))
        self.iona_log_eps = nn.Parameter(torch.tensor(0.0))

    def iona_electrostatic_logits(self, iona_x, iona_dist_ang):
        iona_q = self.iona_charge_head(iona_x)
        iona_qq = iona_q.unsqueeze(2) * iona_q.unsqueeze(1)
        iona_r = iona_dist_ang.unsqueeze(-1).clamp(min=1.0)
        iona_kappa = F.softplus(self.iona_log_kappa)
        iona_eps = F.softplus(self.iona_log_eps) + 1.0
        return iona_qq * torch.exp(-iona_kappa * iona_r / 10.0) / (iona_eps * iona_r)

    def iona_screened_logits(self, iona_R, iona_t, iona_x, iona_z, iona_dist_ang):
        iona_N, iona_L, _ = iona_t.size()
        iona_qn = iona_split_heads(
            self.iona_proj_query(iona_x), self.iona_num_heads, self.iona_hidden_dim
        )
        iona_kn = iona_split_heads(
            self.iona_proj_key(iona_x), self.iona_num_heads, self.iona_hidden_dim
        )
        iona_logits_node = (
            iona_qn.unsqueeze(2)
            * iona_kn.unsqueeze(1)
            * (1 / np.sqrt(self.iona_hidden_dim))
        ).sum(-1)
        iona_logits_pair = self.iona_proj_pair(iona_z)
        iona_qp = iona_split_heads(
            self.iona_proj_query_point(iona_x),
            self.iona_num_heads * self.iona_num_points,
            3,
        )
        iona_qp = iona_local_to_global(iona_R, iona_t, iona_qp).reshape(
            iona_N, iona_L, self.iona_num_heads, -1
        )
        iona_kp = iona_split_heads(
            self.iona_proj_key_point(iona_x),
            self.iona_num_heads * self.iona_num_points,
            3,
        )
        iona_kp = iona_local_to_global(iona_R, iona_t, iona_kp).reshape(
            iona_N, iona_L, self.iona_num_heads, -1
        )
        iona_sq = ((iona_qp.unsqueeze(2) - iona_kp.unsqueeze(1)) ** 2).sum(-1)
        iona_gamma = F.softplus(self.iona_spatial_coef)
        iona_logits_spatial = iona_sq * (
            -1 * iona_gamma * np.sqrt(2 / (9 * self.iona_num_points)) / 2
        )
        iona_logits_elec = torch.nan_to_num(
            self.iona_electrostatic_logits(iona_x, iona_dist_ang)
        )
        return (
            iona_logits_node + iona_logits_pair + iona_logits_spatial + iona_logits_elec
        ) * np.sqrt(1 / 4)

    def forward(
        self,
        iona_R,
        iona_t,
        iona_x,
        iona_z,
        iona_intra_mask,
        iona_inter_mask,
        iona_dist_ang,
    ):
        iona_att_logits = self.iona_screened_logits(
            iona_R, iona_t, iona_x, iona_z, iona_dist_ang
        )
        iona_att_ratio = (
            (
                iona_intra_mask.sum(-1).sum(-1)
                / (
                    iona_intra_mask.sum(-1).sum(-1)
                    + iona_inter_mask.sum(-1).sum(-1)
                    + 1e-08
                )
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
        iona_agg = self.iona_node_aggregation(
            iona_alpha, iona_R, iona_t, iona_x, iona_z
        )
        iona_feats = self.iona_layer_norm_1(
            iona_x + self.iona_mlp_transition_1(iona_agg)
        )
        iona_out = self.iona_mlp_transition_2(iona_feats)
        iona_feats = (
            self.iona_layer_norm_2(iona_feats + iona_out)
            if iona_feats.shape[-1] == iona_out.shape[-1]
            else self.iona_layer_norm_2(iona_out)
        )
        return iona_feats


class IONAScreenedEncoder(IONAGeometricEncoder):

    def __init__(
        self,
        iona_node_feat_dim,
        iona_pair_feat_dim,
        iona_num_layers,
        iona_args,
        iona_codebook_head=1,
    ):
        nn.Module.__init__(self)
        self.iona_blocks = nn.ModuleList(
            [
                IONAScreenedAttentionBlock(
                    iona_node_feat_dim,
                    iona_pair_feat_dim,
                    iona_codebook_head if iona_i == iona_num_layers - 1 else 1,
                )
                for iona_i in range(iona_num_layers)
            ]
        )
        self.iona_args = iona_args

    def forward(self, iona_batch, iona_res_feat, iona_pair_feat):
        iona_pos = iona_batch["pos_atoms"]
        iona_R = iona_construct_3d_basis(
            iona_pos[:, :, IONABackboneAtom.CA],
            iona_pos[:, :, IONABackboneAtom.C],
            iona_pos[:, :, IONABackboneAtom.N],
        )
        iona_t = iona_angstrom_to_nm(iona_pos[:, :, IONABackboneAtom.CA])
        iona_intra_mask, iona_inter_mask = self.iona_build_neighborhood(
            iona_batch,
            iona_cutoff=self.iona_args.iona_cutoff,
            iona_K=self.iona_args.iona_knn,
        )
        iona_ca = iona_pos[:, :, IONABackboneAtom.CA]
        iona_dist_ang = torch.linalg.norm(
            iona_ca[:, :, None] - iona_ca[:, None, :], dim=-1
        )
        for iona_block in self.iona_blocks:
            iona_res_feat = iona_block(
                iona_R,
                iona_t,
                iona_res_feat,
                iona_pair_feat,
                iona_intra_mask,
                iona_inter_mask,
                iona_dist_ang,
            )
        return iona_res_feat


class IONAPredictor(IONABindingPredictor):

    def __init__(self, iona_args):
        super().__init__(iona_args)
        self.iona_attn_encoder = IONAScreenedEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            iona_args.iona_num_layers,
            iona_args,
        )

    def iona_encode(self, iona_batch, iona_vae_model=None):
        iona_h = self.iona_single_encoder(iona_batch)
        iona_z = self.iona_pair_encoder(iona_batch)
        return self.iona_attn_encoder(iona_batch, iona_h, iona_z)
