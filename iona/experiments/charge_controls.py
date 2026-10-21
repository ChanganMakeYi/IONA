import torch
import torch.nn as nn
import torch.nn.functional as F
from iona.experiments.screened_coulomb import (
    IONAScreenedAttentionBlock,
    IONAScreenedEncoder,
    IONAPredictor,
)


class IONAPermutedChargeBlock(IONAScreenedAttentionBlock):

    def iona_electrostatic_logits(self, iona_x, iona_dist_ang):
        iona_q = self.iona_charge_head(iona_x)
        if self.training:
            iona_idx = torch.stack(
                [
                    torch.randperm(iona_q.shape[1], device=iona_q.device)
                    for _ in range(iona_q.shape[0])
                ]
            )
        else:
            iona_g = torch.Generator(device="cpu").manual_seed(0)
            iona_idx = torch.stack(
                [
                    torch.randperm(iona_q.shape[1], generator=iona_g)
                    for _ in range(iona_q.shape[0])
                ]
            ).to(iona_q.device)
        iona_qp = torch.gather(iona_q, 1, iona_idx.unsqueeze(-1).expand_as(iona_q))
        iona_qq = iona_qp.unsqueeze(2) * iona_qp.unsqueeze(1)
        iona_r = iona_dist_ang.unsqueeze(-1).clamp(min=1.0)
        iona_kappa = F.softplus(self.iona_log_kappa)
        iona_eps = F.softplus(self.iona_log_eps) + 1.0
        return iona_qq * torch.exp(-iona_kappa * iona_r / 10.0) / (iona_eps * iona_r)


class IONAUnsignedChargeBlock(IONAScreenedAttentionBlock):

    def iona_electrostatic_logits(self, iona_x, iona_dist_ang):
        iona_q = self.iona_charge_head(iona_x)
        iona_qq = (iona_q.unsqueeze(2) * iona_q.unsqueeze(1)).abs()
        iona_r = iona_dist_ang.unsqueeze(-1).clamp(min=1.0)
        iona_kappa = F.softplus(self.iona_log_kappa)
        iona_eps = F.softplus(self.iona_log_eps) + 1.0
        return iona_qq * torch.exp(-iona_kappa * iona_r / 10.0) / (iona_eps * iona_r)


class IONAUnscreenedBlock(IONAScreenedAttentionBlock):

    def __init__(self, *iona_a, **iona_kw):
        super().__init__(*iona_a, **iona_kw)
        del self.iona_log_kappa

    def iona_electrostatic_logits(self, iona_x, iona_dist_ang):
        iona_q = self.iona_charge_head(iona_x)
        iona_qq = iona_q.unsqueeze(2) * iona_q.unsqueeze(1)
        iona_r = iona_dist_ang.unsqueeze(-1).clamp(min=1.0)
        iona_eps = F.softplus(self.iona_log_eps) + 1.0
        return iona_qq / (iona_eps * iona_r)


def iona_encoder(iona_block_cls):

    class IONAChargeControlEncoder(IONAScreenedEncoder):

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
                    iona_block_cls(
                        iona_node_feat_dim,
                        iona_pair_feat_dim,
                        iona_codebook_head if iona_i == iona_num_layers - 1 else 1,
                    )
                    for iona_i in range(iona_num_layers)
                ]
            )
            self.iona_args = iona_args

    return IONAChargeControlEncoder


def iona_network(iona_block_cls):
    iona_Enc = iona_encoder(iona_block_cls)

    class IONAChargeControlPredictor(IONAPredictor):

        def __init__(self, iona_args):
            super().__init__(iona_args)
            self.iona_attn_encoder = iona_Enc(
                iona_args.iona_node_feat_dim,
                iona_args.iona_pair_feat_dim,
                iona_args.iona_num_layers,
                iona_args,
            )

    return IONAChargeControlPredictor


IONAPermutedChargePredictor = iona_network(IONAPermutedChargeBlock)
IONAUnsignedChargePredictor = iona_network(IONAUnsignedChargeBlock)
IONAUnscreenedPredictor = iona_network(IONAUnscreenedBlock)
