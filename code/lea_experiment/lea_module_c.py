"""LEA-C(四流兼顾): att = (node + pair + spatial + elec) / √4, 无门控, 强制等权。
保留IPA spatial(→Per-Structure) + 加静电elec(→迁移)。elec系数固定=1不可学, 网络不能关掉它。
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
# Portable import bootstrap: prefer $LEAC_CODE, else the sibling code/ dir of this file.
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in [os.environ.get("LEAC_CODE"), os.path.dirname(_here)]:
    if _cand and os.path.isfile(os.path.join(_cand, "module.py")):
        sys.path.insert(0, _cand); break
from module import (PerResidueEncoder, ResiduePairEncoder, GABlock, GAEncoder,
                    _heads, _alpha_from_logits)
from model import DDG_RDE_Network
from common_utils.modules.geometry import local_to_global, construct_3d_basis, angstrom_to_nm
from common_utils.protein.constants import BBHeavyAtom


class LEACGABlock(GABlock):
    def __init__(self, node_feat_dim, pair_feat_dim, codebook_head=1, **kw):
        super().__init__(node_feat_dim, pair_feat_dim, codebook_head, **kw)
        self.charge_head = nn.Linear(node_feat_dim, self.num_heads)
        self.log_kappa = nn.Parameter(torch.tensor(0.0))
        self.log_eps = nn.Parameter(torch.tensor(0.0))

    def elec_logits(self, x, dist_ang):
        q = self.charge_head(x)
        qq = q.unsqueeze(2) * q.unsqueeze(1)
        r = dist_ang.unsqueeze(-1).clamp(min=1.0)
        kappa = F.softplus(self.log_kappa); eps = F.softplus(self.log_eps) + 1.0
        return qq * torch.exp(-kappa * r / 10.0) / (eps * r)

    def attn_logits_C(self, R, t, x, z, dist_ang):
        """node + pair + spatial + elec, 全部/√4, 无门控。"""
        N, L, _ = t.size()
        # node
        qn = _heads(self.proj_query(x), self.num_heads, self.hidden_dim)
        kn = _heads(self.proj_key(x), self.num_heads, self.hidden_dim)
        logits_node = (qn.unsqueeze(2) * kn.unsqueeze(1) * (1 / np.sqrt(self.hidden_dim))).sum(-1)
        # pair
        logits_pair = self.proj_pair(z)
        # spatial (原IPA点注意力, 保留!)
        qp = _heads(self.proj_query_point(x), self.num_heads * self.num_points, 3)
        qp = local_to_global(R, t, qp).reshape(N, L, self.num_heads, -1)
        kp = _heads(self.proj_key_point(x), self.num_heads * self.num_points, 3)
        kp = local_to_global(R, t, kp).reshape(N, L, self.num_heads, -1)
        sq = ((qp.unsqueeze(2) - kp.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sq * ((-1 * gamma * np.sqrt(2 / (9 * self.num_points))) / 2)
        # elec (静电, 长程)
        logits_elec = torch.nan_to_num(self.elec_logits(x, dist_ang))
        return (logits_node + logits_pair + logits_spatial + logits_elec) * np.sqrt(1 / 4)

    def forward(self, R, t, x, z, intra_mask, inter_mask, dist_ang):
        att_logits = self.attn_logits_C(R, t, x, z, dist_ang)
        att_ratio = (intra_mask.sum(-1).sum(-1) / (intra_mask.sum(-1).sum(-1) + inter_mask.sum(-1).sum(-1) + 1e-8)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha = _alpha_from_logits(att_logits, intra_mask) * att_ratio + _alpha_from_logits(att_logits, inter_mask) * (1 - att_ratio)
        agg = self.node_aggregation(alpha, R, t, x, z)   # 聚合含value_point(几何), 保留
        feats = self.layer_norm_1(x + self.mlp_transition_1(agg))
        out = self.mlp_transition_2(feats)
        feats = self.layer_norm_2(feats + out) if feats.shape[-1] == out.shape[-1] else self.layer_norm_2(out)
        return feats


class LEACGAEncoder(GAEncoder):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args, codebook_head=1):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList([
            LEACGABlock(node_feat_dim, pair_feat_dim, codebook_head if i == num_layers-1 else 1)
            for i in range(num_layers)])
        self.args = args

    def forward(self, batch, res_feat, pair_feat):
        pos = batch['pos_atoms']
        R = construct_3d_basis(pos[:, :, BBHeavyAtom.CA], pos[:, :, BBHeavyAtom.C], pos[:, :, BBHeavyAtom.N])
        t = angstrom_to_nm(pos[:, :, BBHeavyAtom.CA])
        intra_mask, inter_mask = self.edge_construction(batch, cutoff=self.args.cutoff, K=self.args.knn)
        ca = pos[:, :, BBHeavyAtom.CA]
        dist_ang = torch.linalg.norm(ca[:, :, None] - ca[:, None, :], dim=-1)
        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, intra_mask, inter_mask, dist_ang)
        return res_feat


class LEAC_DDG_Network(DDG_RDE_Network):
    def __init__(self, args):
        super().__init__(args)
        self.attn_encoder = LEACGAEncoder(args.node_feat_dim, args.pair_feat_dim, args.num_layers, args)

    def encode(self, batch, vae_model=None):
        h = self.single_encoder(batch); z = self.pair_encoder(batch)
        return self.attn_encoder(batch, h, z)
