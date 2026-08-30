"""控制变量架构:
- LEA-G: 第四流 = 距离分桶 per-head 偏置 b_h(bucket(r))(同样全局、非kNN截断,但无电荷物理)。
  用于回答"是屏蔽库仑物理形式在起作用,还是任何 global pairwise bias 都行"。
- LEA-C-gated: 在 LEA-C 静电流上加可学习门控 g,训练后报告 sigmoid(g)。用于定量支撑 g≈0。
所有其它结构与 LEA-C 完全一致(node+pair+spatial+第四流)/√4。
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in [os.environ.get("LEAC_CODE"), os.path.dirname(_here)]:
    if _cand and os.path.isfile(os.path.join(_cand, "module.py")):
        sys.path.insert(0, _cand); break
from module import GABlock, GAEncoder, _heads, _alpha_from_logits
from model import DDG_RDE_Network
from lea_module_c import LEACGABlock  # reuse elec_logits / geometry
from common_utils.modules.geometry import local_to_global, construct_3d_basis, angstrom_to_nm
from common_utils.protein.constants import BBHeavyAtom

NUM_BUCKETS = 64
MAX_DIST = 50.0  # Å


# ---------------- LEA-G: distance-bucketed global bias (control) ----------------
class LEAGGABlock(GABlock):
    def __init__(self, node_feat_dim, pair_feat_dim, codebook_head=1, **kw):
        super().__init__(node_feat_dim, pair_feat_dim, codebook_head, **kw)
        # per-head learnable bias indexed by Cα-Cα distance bucket (global, non-kNN)
        self.dist_bias = nn.Embedding(NUM_BUCKETS, self.num_heads)
        nn.init.zeros_(self.dist_bias.weight)

    def gbias_logits(self, dist_ang):
        buck = torch.clamp((dist_ang / MAX_DIST * NUM_BUCKETS).long(), 0, NUM_BUCKETS - 1)
        return self.dist_bias(buck)  # (N,L,L,num_heads)

    def attn_logits_G(self, R, t, x, z, dist_ang):
        N, L, _ = t.size()
        qn = _heads(self.proj_query(x), self.num_heads, self.hidden_dim)
        kn = _heads(self.proj_key(x), self.num_heads, self.hidden_dim)
        logits_node = (qn.unsqueeze(2) * kn.unsqueeze(1) * (1 / np.sqrt(self.hidden_dim))).sum(-1)
        logits_pair = self.proj_pair(z)
        qp = _heads(self.proj_query_point(x), self.num_heads * self.num_points, 3)
        qp = local_to_global(R, t, qp).reshape(N, L, self.num_heads, -1)
        kp = _heads(self.proj_key_point(x), self.num_heads * self.num_points, 3)
        kp = local_to_global(R, t, kp).reshape(N, L, self.num_heads, -1)
        sq = ((qp.unsqueeze(2) - kp.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sq * ((-1 * gamma * np.sqrt(2 / (9 * self.num_points))) / 2)
        logits_g = torch.nan_to_num(self.gbias_logits(dist_ang))
        return (logits_node + logits_pair + logits_spatial + logits_g) * np.sqrt(1 / 4)

    def forward(self, R, t, x, z, intra_mask, inter_mask, dist_ang):
        att_logits = self.attn_logits_G(R, t, x, z, dist_ang)
        att_ratio = (intra_mask.sum(-1).sum(-1) / (intra_mask.sum(-1).sum(-1) + inter_mask.sum(-1).sum(-1) + 1e-8)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha = _alpha_from_logits(att_logits, intra_mask) * att_ratio + _alpha_from_logits(att_logits, inter_mask) * (1 - att_ratio)
        agg = self.node_aggregation(alpha, R, t, x, z)
        feats = self.layer_norm_1(x + self.mlp_transition_1(agg))
        out = self.mlp_transition_2(feats)
        feats = self.layer_norm_2(feats + out) if feats.shape[-1] == out.shape[-1] else self.layer_norm_2(out)
        return feats


# ---------------- LEA-C-gated: learnable gate on elec stream (for g readout) ------
class LEACGatedGABlock(LEACGABlock):
    def __init__(self, node_feat_dim, pair_feat_dim, codebook_head=1, **kw):
        super().__init__(node_feat_dim, pair_feat_dim, codebook_head, **kw)
        self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 init

    def attn_logits_C(self, R, t, x, z, dist_ang):
        N, L, _ = t.size()
        qn = _heads(self.proj_query(x), self.num_heads, self.hidden_dim)
        kn = _heads(self.proj_key(x), self.num_heads, self.hidden_dim)
        logits_node = (qn.unsqueeze(2) * kn.unsqueeze(1) * (1 / np.sqrt(self.hidden_dim))).sum(-1)
        logits_pair = self.proj_pair(z)
        qp = _heads(self.proj_query_point(x), self.num_heads * self.num_points, 3)
        qp = local_to_global(R, t, qp).reshape(N, L, self.num_heads, -1)
        kp = _heads(self.proj_key_point(x), self.num_heads * self.num_points, 3)
        kp = local_to_global(R, t, kp).reshape(N, L, self.num_heads, -1)
        sq = ((qp.unsqueeze(2) - kp.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sq * ((-1 * gamma * np.sqrt(2 / (9 * self.num_points))) / 2)
        logits_elec = torch.sigmoid(self.gate) * torch.nan_to_num(self.elec_logits(x, dist_ang))
        return (logits_node + logits_pair + logits_spatial + logits_elec) * np.sqrt(1 / 4)


class _LEAEncoder(GAEncoder):
    BLOCK = None
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args, codebook_head=1):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList([
            self.BLOCK(node_feat_dim, pair_feat_dim, codebook_head if i == num_layers - 1 else 1)
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

class LEAGGAEncoder(_LEAEncoder):     BLOCK = LEAGGABlock
class LEACGatedGAEncoder(_LEAEncoder): BLOCK = LEACGatedGABlock

class LEAG_DDG_Network(DDG_RDE_Network):
    def __init__(self, args):
        super().__init__(args)
        self.attn_encoder = LEAGGAEncoder(args.node_feat_dim, args.pair_feat_dim, args.num_layers, args)
    def encode(self, batch, vae_model=None):
        return self.attn_encoder(batch, self.single_encoder(batch), self.pair_encoder(batch))

class LEACGated_DDG_Network(DDG_RDE_Network):
    def __init__(self, args):
        super().__init__(args)
        self.attn_encoder = LEACGatedGAEncoder(args.node_feat_dim, args.pair_feat_dim, args.num_layers, args)
    def encode(self, batch, vae_model=None):
        return self.attn_encoder(batch, self.single_encoder(batch), self.pair_encoder(batch))
