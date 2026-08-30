"""LEA-D = LEA-C plus an explicit Born-type desolvation penalty.

att = (node + pair + spatial + elec + desolv) / sqrt(5), no gating, equal weights.

Motivation. Pretraining regresses the APBS Poisson-Boltzmann interfacial potential, which
contains BOTH the direct Coulomb interaction AND the desolvation (Born) cost of moving
charge from water into a low-dielectric interface. LEA-C's attention bias encodes only the
first half, so the architecture is less expressive than its own pretraining signal.

That gap matches a measured failure. On SKEMPI's protease-inhibitor subset the electrostatic
stream contributes nothing (delta -0.001, n=1349) even though charge matters MORE there than
anywhere else energetically (|ddG| 3.01 for charge-changing versus 1.86 for neutral;
corr(|ddG|, charge-changing) = 0.241, the highest of any complex class). Those interfaces are
rigid and deeply buried, so the dominant electrostatic term is the cost of burying a charge,
not the attraction between two of them; and burial is already recoverable from IPA geometry,
which is why a pairwise Coulomb term added nothing there.

Three constraints keep this a physics claim rather than extra capacity:

  1. REUSES LEACGABlock's charge head. Not a new free variable. Both terms depend on burial,
     so independent charge parameters could trade off arbitrarily and neither would be
     attributable -- destroying the property that makes the matched ablation interpretable.
  2. Sign fixed negative via -softplus. Burying an unpaired charge can only be penalised. A
     freely signed term degenerates into a generic learned pairwise channel, which is exactly
     the "any extra bias would do" objection.
  3. Uses q_i^2 with a SHORT-RANGE burial kernel, against the Coulomb term's q_i q_j with a
     screened 1/r tail: different functional form and different range, hence separable.

Prior evidence to keep in mind: adding SASA to the PRETRAINING objective at 12M did not help
(SKEMPI per-structure 0.4515 for SASA-only and 0.4589 for both, versus 0.464 for
electrostatic alone). That was a pretraining-objective change rather than an architectural
term, but it is a reason to expect this may come out flat or negative -- which is exactly why
it is being tested at 12M first.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lea_module_c import LEACGABlock, LEACGAEncoder, LEAC_DDG_Network
from module import _alpha_from_logits, _heads
from common_utils.modules.geometry import local_to_global


class LEADGABlock(LEACGABlock):
    def __init__(self, node_feat_dim, pair_feat_dim, codebook_head=1, **kw):
        super().__init__(node_feat_dim, pair_feat_dim, codebook_head, **kw)
        # magnitude only; the sign is hard-coded negative in desolv_logits
        self.log_lambda_d = nn.Parameter(torch.tensor(0.0))
        # burial-kernel width in Angstrom, kept positive
        self.log_r0 = nn.Parameter(torch.tensor(float(np.log(np.exp(6.0) - 1.0))))

    def desolv_logits(self, q, r):
        """Born self-energy q_i^2 weighted by how much neighbour j buries i.

        Short-ranged and strictly non-positive: it can only penalise burying charge.
        q is passed in from the Coulomb term so the two streams share one charge head.
        """
        lam = F.softplus(self.log_lambda_d)
        r0 = F.softplus(self.log_r0) + 1e-3
        w = torch.exp(-(r / r0) ** 2)                       # (N, L, L, 1) short range
        q2 = (q ** 2).unsqueeze(2)                          # (N, L, 1, H) self-energy
        return -lam * q2 * w

    def attn_logits_D(self, R, t, x, z, dist_ang):
        """node + pair + spatial + elec + desolv, all / sqrt(5), no gating."""
        N, L, _ = t.size()
        qn = _heads(self.proj_query(x), self.num_heads, self.hidden_dim)
        kn = _heads(self.proj_key(x), self.num_heads, self.hidden_dim)
        logits_node = (qn.unsqueeze(2) * kn.unsqueeze(1)
                       * (1 / np.sqrt(self.hidden_dim))).sum(-1)
        logits_pair = self.proj_pair(z)
        qp = _heads(self.proj_query_point(x), self.num_heads * self.num_points, 3)
        qp = local_to_global(R, t, qp).reshape(N, L, self.num_heads, -1)
        kp = _heads(self.proj_key_point(x), self.num_heads * self.num_points, 3)
        kp = local_to_global(R, t, kp).reshape(N, L, self.num_heads, -1)
        sq = ((qp.unsqueeze(2) - kp.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sq * ((-1 * gamma * np.sqrt(2 / (9 * self.num_points))) / 2)

        # the two physical streams, sharing one set of learned charges
        q = self.charge_head(x)
        r = dist_ang.unsqueeze(-1).clamp(min=1.0)
        kappa = F.softplus(self.log_kappa)
        eps = F.softplus(self.log_eps) + 1.0
        qq = q.unsqueeze(2) * q.unsqueeze(1)
        logits_elec = torch.nan_to_num(qq * torch.exp(-kappa * r / 10.0) / (eps * r))
        logits_desolv = torch.nan_to_num(self.desolv_logits(q, r))

        return (logits_node + logits_pair + logits_spatial
                + logits_elec + logits_desolv) * np.sqrt(1 / 5)

    def forward(self, R, t, x, z, intra_mask, inter_mask, dist_ang):
        att_logits = self.attn_logits_D(R, t, x, z, dist_ang)
        att_ratio = (intra_mask.sum(-1).sum(-1)
                     / (intra_mask.sum(-1).sum(-1) + inter_mask.sum(-1).sum(-1) + 1e-8)
                     ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha = (_alpha_from_logits(att_logits, intra_mask) * att_ratio
                 + _alpha_from_logits(att_logits, inter_mask) * (1 - att_ratio))
        agg = self.node_aggregation(alpha, R, t, x, z)
        feats = self.layer_norm_1(x + self.mlp_transition_1(agg))
        out = self.mlp_transition_2(feats)
        feats = (self.layer_norm_2(feats + out) if feats.shape[-1] == out.shape[-1]
                 else self.layer_norm_2(out))
        return feats


class LEADGAEncoder(LEACGAEncoder):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args, codebook_head=1):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList([
            LEADGABlock(node_feat_dim, pair_feat_dim,
                        codebook_head if i == num_layers - 1 else 1)
            for i in range(num_layers)])
        self.args = args


class LEAD_DDG_Network(LEAC_DDG_Network):
    def __init__(self, args):
        super().__init__(args)
        self.attn_encoder = LEADGAEncoder(args.node_feat_dim, args.pair_feat_dim,
                                          args.num_layers, args)
