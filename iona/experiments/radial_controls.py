import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from iona.experiments.screened_coulomb import (
    IONAScreenedAttentionBlock,
    IONAScreenedEncoder,
    IONAPredictor,
)


class IONARadialProfile(nn.Module):

    def __init__(
        self, iona_num_heads, iona_num_basis=16, iona_hidden=32, iona_r_max=32.0
    ):
        super().__init__()
        self.register_buffer(
            "iona_centres", torch.linspace(0.0, iona_r_max, iona_num_basis)
        )
        self.register_buffer(
            "iona_width", torch.tensor(iona_r_max / max(iona_num_basis - 1, 1))
        )
        self.iona_net = nn.Sequential(
            nn.Linear(iona_num_basis, iona_hidden),
            nn.SiLU(),
            nn.Linear(iona_hidden, iona_num_heads),
        )

    def forward(self, iona_r):
        iona_rbf = torch.exp(
            -(((iona_r.unsqueeze(-1) - self.iona_centres) / self.iona_width) ** 2)
        )
        return self.iona_net(iona_rbf)


class IONALearnedRadialBlock(IONAScreenedAttentionBlock):

    def __init__(
        self,
        iona_node_feat_dim,
        iona_pair_feat_dim,
        iona_codebook_head=1,
        iona_keep_scalars=True,
        **iona_kw
    ):
        super().__init__(
            iona_node_feat_dim, iona_pair_feat_dim, iona_codebook_head, **iona_kw
        )
        self.iona_keep_scalars = iona_keep_scalars
        self.iona_radial = IONARadialProfile(self.iona_num_heads)
        if not iona_keep_scalars:
            del self.iona_charge_head
        del self.iona_log_kappa
        del self.iona_log_eps

    def iona_electrostatic_logits(self, iona_x, iona_dist_ang):
        iona_r = iona_dist_ang.clamp(min=1.0)
        iona_f = self.iona_radial(iona_r)
        if not self.iona_keep_scalars:
            return iona_f
        iona_q = self.iona_charge_head(iona_x)
        return iona_q.unsqueeze(2) * iona_q.unsqueeze(1) * iona_f


class IONALearnedRadialEncoder(IONAScreenedEncoder):

    def __init__(
        self,
        iona_node_feat_dim,
        iona_pair_feat_dim,
        iona_num_layers,
        iona_args,
        iona_codebook_head=1,
        iona_keep_scalars=True,
    ):
        nn.Module.__init__(self)
        self.iona_blocks = nn.ModuleList(
            [
                IONALearnedRadialBlock(
                    iona_node_feat_dim,
                    iona_pair_feat_dim,
                    iona_codebook_head if iona_i == iona_num_layers - 1 else 1,
                    iona_keep_scalars=iona_keep_scalars,
                )
                for iona_i in range(iona_num_layers)
            ]
        )
        self.iona_args = iona_args


class IONALearnedRadialPredictor(IONAPredictor):
    iona_KEEP_SCALARS = True

    def __init__(self, iona_args):
        super().__init__(iona_args)
        self.iona_attn_encoder = IONALearnedRadialEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            iona_args.iona_num_layers,
            iona_args,
            iona_keep_scalars=self.iona_KEEP_SCALARS,
        )


class IONADistanceOnlyPredictor(IONALearnedRadialPredictor):
    iona_KEEP_SCALARS = False
