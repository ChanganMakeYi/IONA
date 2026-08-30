"""B1. Functional-form controls for the screened-Coulomb stream.

Reading the trained LEA-C stream showed that its learned per-residue scalars do not track formal
charge, that the pairwise bias does not reproduce a formal-charge Coulomb interaction, and that
most of its rank variation is radial. That raises the obvious question the paper cannot currently
answer: is the Coulomb functional form doing the work, or would any pairwise radial channel of
matched capacity do as well?

Two controls are defined here. Both keep everything else identical to LEA-C -- the same insertion
point, the same equal-weight gate-free fusion by 1/sqrt(4), the same three geometric streams, the
same training protocol -- so that the only thing varying is the piece under test.

  LEA-R  (arch `lear`)   The per-residue scalar path is untouched: q_i still comes from a linear
                         projection of node features, and the bias is still the bilinear product
                         q_i q_j. Only the RADIAL PROFILE changes, from the Coulomb-shaped
                         exp(-kappa r/10)/(eps r) to an unrestricted function of distance learned
                         through a small MLP over Gaussian radial basis features. If LEA-R matches
                         LEA-C, the Coulomb shape is not what carries the benefit. This is the
                         tighter of the two controls, because it varies one thing.

  LEA-D  (arch `leard`)  The radial profile is learned as in LEA-R, but the per-residue scalars
                         are removed entirely, leaving a pure learned distance kernel shared
                         across residue pairs within a head. If LEA-D also matches, then neither
                         the Coulomb shape nor the latent scalars are load-bearing, and the
                         benefit is attributable to having a pairwise radial channel at all.

The parameter cost of the radial MLP is small and is reported rather than hidden: with 16 basis
functions and a hidden width of 32 it adds 16*32 + 32 + 32*H + H weights per block against the
1024*H + H + 2 of LEA-C, so LEA-R is slightly LARGER than LEA-C. Any advantage LEA-C shows over
LEA-R therefore cannot be explained by capacity, which is the direction that matters for the
claim being tested.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lea_module_c import LEACGABlock, LEACGAEncoder, LEAC_DDG_Network


class RadialProfile(nn.Module):
    """An unrestricted learned function of distance, per attention head.

    Distances are expanded on a fixed grid of Gaussians spanning the range over which the
    geometric streams operate, and an MLP maps that expansion to one value per head. Nothing
    constrains the result to decay, to be positive, or to resemble a screened Coulomb term; that
    is the point of the control.
    """

    def __init__(self, num_heads, num_basis=16, hidden=32, r_max=32.0):
        super().__init__()
        self.register_buffer("centres", torch.linspace(0.0, r_max, num_basis))
        self.register_buffer("width", torch.tensor(r_max / max(num_basis - 1, 1)))
        self.net = nn.Sequential(nn.Linear(num_basis, hidden), nn.SiLU(),
                                 nn.Linear(hidden, num_heads))

    def forward(self, r):
        # r: (B, L, L) in angstrom -> (B, L, L, H)
        rbf = torch.exp(-((r.unsqueeze(-1) - self.centres) / self.width) ** 2)
        return self.net(rbf)


class LEARGABlock(LEACGABlock):
    """LEA-C with the Coulomb radial profile replaced by a learned one."""

    def __init__(self, node_feat_dim, pair_feat_dim, codebook_head=1, keep_scalars=True, **kw):
        super().__init__(node_feat_dim, pair_feat_dim, codebook_head, **kw)
        self.keep_scalars = keep_scalars
        self.radial = RadialProfile(self.num_heads)
        if not keep_scalars:
            # the distance-only control has no per-residue scalar path at all
            del self.charge_head
        # kappa and eps are inherited but unused; drop them so the checkpoint cannot mislead
        del self.log_kappa
        del self.log_eps

    def elec_logits(self, x, dist_ang):
        r = dist_ang.clamp(min=1.0)
        f = self.radial(r)                                  # (B, L, L, H)
        if not self.keep_scalars:
            return f
        q = self.charge_head(x)                             # (B, L, H)
        return (q.unsqueeze(2) * q.unsqueeze(1)) * f


class LEARGAEncoder(LEACGAEncoder):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args, codebook_head=1,
                 keep_scalars=True):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList([
            LEARGABlock(node_feat_dim, pair_feat_dim,
                        codebook_head if i == num_layers - 1 else 1,
                        keep_scalars=keep_scalars)
            for i in range(num_layers)])
        self.args = args


class LEAR_DDG_Network(LEAC_DDG_Network):
    """Learned radial profile, per-residue scalars retained."""
    KEEP_SCALARS = True

    def __init__(self, args):
        super().__init__(args)
        self.attn_encoder = LEARGAEncoder(args.node_feat_dim, args.pair_feat_dim,
                                          args.num_layers, args,
                                          keep_scalars=self.KEEP_SCALARS)


class LEAD_RADIAL_DDG_Network(LEAR_DDG_Network):
    """Learned radial profile, no per-residue scalars: a pure distance kernel."""
    KEEP_SCALARS = False
