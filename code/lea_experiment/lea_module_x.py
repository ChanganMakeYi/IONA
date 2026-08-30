"""Three further functional-form controls for the screened-Coulomb stream.

The Results section names five controls that would isolate what the Coulomb form contributes.
Two are implemented in `lea_module_r` (a learned radial profile, with and without the per-residue
scalars). The remaining three are here. Each changes exactly one property of

    logits_elec_ij = q_i q_j exp(-kappa r_ij / 10) / (eps r_ij)

and leaves the insertion point, the equal-weight gate-free fusion, the three geometric streams,
the pretrained initialisation and the training protocol identical to LEA-C.

  LEA-P (`leap`)   Permuted scalars. The per-residue scalars are computed exactly as in LEA-C and
                   then randomly permuted across residues within each structure before the outer
                   product is formed. The distribution of q is preserved; only the assignment of a
                   scalar to a residue is destroyed. If LEA-P matches LEA-C, the benefit does not
                   depend on which residue carries which scalar, and the stream is acting through
                   its kernel rather than through learned per-residue modulation. The permutation
                   is redrawn every forward pass, so the network cannot learn around a fixed one.

  LEA-A (`leaa`)   Sign-free interaction, |q_i q_j| in place of q_i q_j. An electrostatic term
                   distinguishes like from unlike charge; a magnitude-only term cannot. If LEA-A
                   matches LEA-C, the sign structure that makes the term electrostatic in
                   character is not what carries the benefit.

  LEA-U (`leau`)   Unscreened, q_i q_j / (eps r) with the exponential removed. This tests the
                   screening specifically, holding the 1/r decay and the bilinear scalar
                   structure fixed. kappa is deleted rather than set to zero so that a checkpoint
                   cannot silently carry a stale value.

None of the three adds parameters beyond LEA-C except LEA-U, which removes one. Any difference
therefore cannot be attributed to capacity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lea_module_c import LEACGABlock, LEACGAEncoder, LEAC_DDG_Network


class LEAPGABlock(LEACGABlock):
    """Per-residue scalars permuted across residues, redrawn every forward pass."""

    def elec_logits(self, x, dist_ang):
        q = self.charge_head(x)                                  # (B, L, H)
        if self.training:
            idx = torch.stack([torch.randperm(q.shape[1], device=q.device)
                               for _ in range(q.shape[0])])      # (B, L)
        else:
            # a fixed permutation at evaluation, so scoring is deterministic
            g = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.stack([torch.randperm(q.shape[1], generator=g)
                               for _ in range(q.shape[0])]).to(q.device)
        qp = torch.gather(q, 1, idx.unsqueeze(-1).expand_as(q))
        qq = qp.unsqueeze(2) * qp.unsqueeze(1)
        r = dist_ang.unsqueeze(-1).clamp(min=1.0)
        kappa = F.softplus(self.log_kappa)
        eps = F.softplus(self.log_eps) + 1.0
        return qq * torch.exp(-kappa * r / 10.0) / (eps * r)


class LEAAGABlock(LEACGABlock):
    """Sign-free: the magnitude of the scalar product, with no like/unlike distinction."""

    def elec_logits(self, x, dist_ang):
        q = self.charge_head(x)
        qq = (q.unsqueeze(2) * q.unsqueeze(1)).abs()
        r = dist_ang.unsqueeze(-1).clamp(min=1.0)
        kappa = F.softplus(self.log_kappa)
        eps = F.softplus(self.log_eps) + 1.0
        return qq * torch.exp(-kappa * r / 10.0) / (eps * r)


class LEAUGABlock(LEACGABlock):
    """Unscreened Coulomb: the exponential screening factor removed, 1/r retained."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        del self.log_kappa                     # no screening length to learn or to inherit

    def elec_logits(self, x, dist_ang):
        q = self.charge_head(x)
        qq = q.unsqueeze(2) * q.unsqueeze(1)
        r = dist_ang.unsqueeze(-1).clamp(min=1.0)
        eps = F.softplus(self.log_eps) + 1.0
        return qq / (eps * r)


def _encoder(block_cls):
    class _E(LEACGAEncoder):
        def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args, codebook_head=1):
            nn.Module.__init__(self)
            self.blocks = nn.ModuleList([
                block_cls(node_feat_dim, pair_feat_dim,
                          codebook_head if i == num_layers - 1 else 1)
                for i in range(num_layers)])
            self.args = args
    return _E


def _network(block_cls):
    Enc = _encoder(block_cls)

    class _N(LEAC_DDG_Network):
        def __init__(self, args):
            super().__init__(args)
            self.attn_encoder = Enc(args.node_feat_dim, args.pair_feat_dim,
                                    args.num_layers, args)
    return _N


LEAP_DDG_Network = _network(LEAPGABlock)      # permuted scalars
LEAA_DDG_Network = _network(LEAAGABlock)      # sign-free
LEAU_DDG_Network = _network(LEAUGABlock)      # unscreened
