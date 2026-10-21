import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from iona.experiments.screened_coulomb import (
    IONAScreenedAttentionBlock,
    IONAScreenedEncoder,
    IONAPredictor,
)
from iona.encoders import iona_masked_attention, iona_split_heads
from iona.primitives.modules.geometry import iona_local_to_global


class IONASolvationAttentionBlock(IONAScreenedAttentionBlock):

    def __init__(
        self, iona_node_feat_dim, iona_pair_feat_dim, iona_codebook_head=1, **iona_kw
    ):
        super().__init__(
            iona_node_feat_dim, iona_pair_feat_dim, iona_codebook_head, **iona_kw
        )
        self.iona_log_lambda_d = nn.Parameter(torch.tensor(0.0))
        self.iona_log_r0 = nn.Parameter(torch.tensor(float(np.log(np.exp(6.0) - 1.0))))

    def iona_desolv_logits(self, iona_q, iona_r):
        iona_lam = F.softplus(self.iona_log_lambda_d)
        iona_r0 = F.softplus(self.iona_log_r0) + 0.001
        iona_w = torch.exp(-((iona_r / iona_r0) ** 2))
        iona_q2 = (iona_q**2).unsqueeze(2)
        return -iona_lam * iona_q2 * iona_w

    def iona_solvation_logits(self, iona_R, iona_t, iona_x, iona_z, iona_dist_ang):
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
        iona_q = self.iona_charge_head(iona_x)
        iona_r = iona_dist_ang.unsqueeze(-1).clamp(min=1.0)
        iona_kappa = F.softplus(self.iona_log_kappa)
        iona_eps = F.softplus(self.iona_log_eps) + 1.0
        iona_qq = iona_q.unsqueeze(2) * iona_q.unsqueeze(1)
        iona_logits_elec = torch.nan_to_num(
            iona_qq * torch.exp(-iona_kappa * iona_r / 10.0) / (iona_eps * iona_r)
        )
        iona_logits_desolv = torch.nan_to_num(self.iona_desolv_logits(iona_q, iona_r))
        return (
            iona_logits_node
            + iona_logits_pair
            + iona_logits_spatial
            + iona_logits_elec
            + iona_logits_desolv
        ) * np.sqrt(1 / 5)

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
        iona_att_logits = self.iona_solvation_logits(
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


class IONASolvationEncoder(IONAScreenedEncoder):

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
                IONASolvationAttentionBlock(
                    iona_node_feat_dim,
                    iona_pair_feat_dim,
                    iona_codebook_head if iona_i == iona_num_layers - 1 else 1,
                )
                for iona_i in range(iona_num_layers)
            ]
        )
        self.iona_args = iona_args


class IONASolvationPredictor(IONAPredictor):

    def __init__(self, iona_args):
        super().__init__(iona_args)
        self.iona_attn_encoder = IONASolvationEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            iona_args.iona_num_layers,
            iona_args,
        )
