from iona.checkpoint_io import IONACheckpointAdapter
import torch
import torch.nn as nn
import torch.nn.functional as F
from iona.encoders import (
    IONAResidueEncoder,
    IONAPairEncoder,
    IONAGeometricEncoder,
    IONAGeometricDecoder,
)

iona_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IONABindingPredictor(IONACheckpointAdapter, nn.Module):

    def __init__(self, iona_args):
        super().__init__()
        self.iona_args = iona_args
        iona_input_dim = iona_args.iona_node_feat_dim
        self.iona_single_encoder = IONAResidueEncoder(
            iona_feat_dim=iona_args.iona_node_feat_dim, iona_args=iona_args
        )
        self.iona_pair_encoder = IONAPairEncoder(
            iona_feat_dim=iona_args.iona_pair_feat_dim
        )
        self.iona_attn_encoder = IONAGeometricEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            iona_args.iona_num_layers,
            iona_args,
        )
        self.iona_ddg_readout = nn.Sequential(
            nn.Linear(iona_input_dim, iona_args.iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_args.iona_node_feat_dim, iona_args.iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_args.iona_node_feat_dim, 1),
        )
        self.iona_single_fusion = nn.Sequential(
            nn.Linear(iona_args.iona_node_feat_dim, iona_args.iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_args.iona_node_feat_dim, iona_args.iona_node_feat_dim),
        )
        self.iona_attn_weight = nn.Linear(3 * iona_args.iona_node_feat_dim, 3)

    def iona_encode(self, iona_batch, iona_vae_model):
        iona_h = self.iona_single_encoder(iona_batch)
        if iona_vae_model is not None:
            iona_e_list = iona_vae_model.iona_forward_encoder(iona_batch)
            iona_weight = self.iona_attn_weight(
                torch.cat(iona_e_list, dim=-1)
            ).unsqueeze(dim=-2)
            iona_e = (
                iona_weight
                * torch.cat(
                    [
                        iona_e_list[0].unsqueeze(dim=-1),
                        iona_e_list[1].unsqueeze(dim=-1),
                        iona_e_list[2].unsqueeze(dim=-1),
                    ],
                    dim=-1,
                )
            ).sum(dim=-1)
            iona_intra_mask, iona_inter_mask = (
                self.iona_attn_encoder.iona_build_neighborhood(
                    iona_batch,
                    iona_cutoff=self.iona_args.iona_cutoff,
                    iona_K=self.iona_args.iona_knn,
                )
            )
            iona_mask = iona_intra_mask + iona_inter_mask
            iona_mask = (
                (iona_mask * iona_batch["mut_flag"][:, None, :]).float().detach()
            )
            iona_eq = torch.einsum("ijk,ikn->ijn", iona_mask, iona_e) / (
                torch.sum(iona_mask, dim=-1, keepdim=True) + 1e-06
            )
            iona_h = self.iona_single_fusion(iona_h + iona_eq)
        iona_z = self.iona_pair_encoder(iona_batch)
        iona_h = self.iona_attn_encoder(iona_batch, iona_h, iona_z)
        return iona_h

    def forward(self, iona_batch, iona_vae_mode=None):
        iona_batch_wt = {iona_k: iona_v for iona_k, iona_v in iona_batch.items()}
        iona_batch_mt = {iona_k: iona_v for iona_k, iona_v in iona_batch.items()}
        iona_batch_mt["aa"] = iona_batch_mt["aa_mut"]
        iona_h_wt = self.iona_encode(iona_batch_wt, iona_vae_mode)
        iona_h_mt = self.iona_encode(iona_batch_mt, iona_vae_mode)
        iona_H_mt, iona_H_wt = (iona_h_mt.max(dim=1)[0], iona_h_wt.max(dim=1)[0])
        iona_ddg_pred = self.iona_ddg_readout(iona_H_mt - iona_H_wt).squeeze(-1)
        iona_loss = F.mse_loss(iona_ddg_pred, iona_batch["ddG"])
        iona_out_dict = {"ddG_pred": iona_ddg_pred, "ddG_true": iona_batch["ddG"]}
        return (iona_loss, iona_out_dict)


class IONARepresentationCodebook(IONACheckpointAdapter, nn.Module):

    def __init__(self, iona_args):
        super().__init__()
        self.iona_single_encoder = IONAResidueEncoder(
            iona_feat_dim=iona_args.iona_node_feat_dim, iona_args=iona_args
        )
        self.iona_pair_encoder = IONAPairEncoder(
            iona_feat_dim=iona_args.iona_pair_feat_dim
        )
        self.iona_attn_encoder = IONAGeometricEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            iona_args.iona_num_layers,
            iona_args,
            iona_codebook_head=3,
        )
        self.iona_vq_layer_type = IONAVectorQuantizer(
            iona_args.iona_node_feat_dim,
            iona_args.iona_num_embeddings,
            iona_args.iona_commitment_cost,
        )
        self.iona_vq_layer_angle = IONAVectorQuantizer(
            iona_args.iona_node_feat_dim,
            iona_args.iona_num_embeddings,
            iona_args.iona_commitment_cost,
        )
        self.iona_vq_layer_coord = IONAVectorQuantizer(
            iona_args.iona_node_feat_dim,
            iona_args.iona_num_embeddings,
            iona_args.iona_commitment_cost,
        )
        self.iona_pair_decoder = IONAPairEncoder(
            iona_feat_dim=iona_args.iona_pair_feat_dim
        )
        self.iona_attn_decoder_type = IONAGeometricEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            int(iona_args.iona_num_layers / 2),
            iona_args,
        )
        self.iona_attn_decoder_angle = IONAGeometricEncoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            int(iona_args.iona_num_layers / 2),
            iona_args,
        )
        self.iona_attn_decoder_coord = IONAGeometricDecoder(
            iona_args.iona_node_feat_dim,
            iona_args.iona_pair_feat_dim,
            int(iona_args.iona_num_layers / 2),
            iona_args,
        )
        self.iona_aa_type_decoder = nn.Sequential(
            nn.Linear(iona_args.iona_node_feat_dim, iona_args.iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_args.iona_node_feat_dim, 22),
        )
        self.iona_aa_angle_decoder = nn.Sequential(
            nn.Linear(iona_args.iona_node_feat_dim, iona_args.iona_node_feat_dim),
            nn.ReLU(),
            nn.Linear(iona_args.iona_node_feat_dim, 7 * 13),
        )
        self.iona_args = iona_args

    def iona_encode(self, iona_batch):
        iona_h, iona_gt_mask = self.iona_single_encoder(iona_batch, iona_pretrain=True)
        iona_z = self.iona_pair_encoder(iona_batch)
        iona_h = self.iona_attn_encoder(iona_batch, iona_h, iona_z)
        return (iona_h, iona_gt_mask)

    def iona_decode(self, iona_batch, iona_e_type, iona_e_angle, iona_e_coord):
        iona_z = self.iona_pair_decoder(iona_batch)
        iona_h_type = self.iona_aa_type_decoder(
            self.iona_attn_decoder_type(iona_batch, iona_e_type, iona_z)
        )
        iona_h_angle = self.iona_aa_angle_decoder(
            self.iona_attn_decoder_angle(iona_batch, iona_e_angle, iona_z)
        )
        iona_h_coord, iona_mask_coord = self.iona_attn_decoder_coord(
            iona_batch, iona_e_coord, iona_z
        )
        return (iona_h_type, iona_h_angle, iona_h_coord, iona_mask_coord)

    def forward(self, iona_batch):
        iona_batch_wt = {iona_k: iona_v for iona_k, iona_v in iona_batch.items()}
        iona_h_wt, iona_gt_mask = self.iona_encode(iona_batch_wt)
        iona_e_type, iona_e_q_loss_type = self.iona_vq_layer_type(
            iona_h_wt[:, :, : self.iona_args.iona_node_feat_dim],
            iona_batch["mask_atoms"][:, :, 1],
        )
        iona_e_angle, iona_e_q_loss_angle = self.iona_vq_layer_angle(
            iona_h_wt[
                :,
                :,
                self.iona_args.iona_node_feat_dim : 2
                * self.iona_args.iona_node_feat_dim,
            ],
            iona_batch["mask_atoms"][:, :, 1],
        )
        iona_e_coord, iona_e_q_loss_coord = self.iona_vq_layer_coord(
            iona_h_wt[:, :, 2 * self.iona_args.iona_node_feat_dim :],
            iona_batch["mask_atoms"][:, :, 1],
        )
        iona_e_q_loss = iona_e_q_loss_type + iona_e_q_loss_angle + iona_e_q_loss_coord
        iona_h_type, iona_h_angle, iona_h_coord, iona_mask_coord = self.iona_decode(
            iona_batch_wt, iona_e_type, iona_e_angle, iona_e_coord
        )
        iona_s_recon_loss = F.cross_entropy(
            iona_h_type.reshape(-1, 22),
            iona_batch_wt["aa"].reshape(-1),
            reduction="none",
        )
        iona_s_recon_loss = (
            torch.sum(iona_s_recon_loss[iona_gt_mask[1].reshape(-1)])
            / iona_gt_mask[1].sum()
        )
        iona_h_recon_loss = F.mse_loss(
            iona_h_angle[iona_gt_mask[2]],
            iona_gt_mask[0][iona_gt_mask[2]],
            reduction="mean",
        )
        iona_x_recon_loss = F.smooth_l1_loss(
            iona_h_coord[iona_mask_coord],
            iona_batch["pos_atoms"][:, :, :4][iona_mask_coord],
            reduction="mean",
        )
        return (
            [iona_e_type.detach(), iona_e_angle.detach(), iona_e_coord.detach()],
            iona_e_q_loss,
            iona_s_recon_loss,
            iona_h_recon_loss,
            iona_x_recon_loss,
        )

    def iona_forward_encoder(self, iona_batch):
        iona_batch_wt = {iona_k: iona_v for iona_k, iona_v in iona_batch.items()}
        iona_h = self.iona_single_encoder(iona_batch_wt, iona_pretrain=False)
        iona_z = self.iona_pair_encoder(iona_batch_wt)
        iona_h_wt = self.iona_attn_encoder(iona_batch_wt, iona_h, iona_z)
        iona_e_type, _ = self.iona_vq_layer_type(
            iona_h_wt[:, :, : self.iona_args.iona_node_feat_dim],
            iona_batch["mask_atoms"][:, :, 1],
        )
        iona_e_angle, _ = self.iona_vq_layer_angle(
            iona_h_wt[
                :,
                :,
                self.iona_args.iona_node_feat_dim : 2
                * self.iona_args.iona_node_feat_dim,
            ],
            iona_batch["mask_atoms"][:, :, 1],
        )
        iona_e_coord, _ = self.iona_vq_layer_coord(
            iona_h_wt[:, :, 2 * self.iona_args.iona_node_feat_dim :],
            iona_batch["mask_atoms"][:, :, 1],
        )
        return [iona_e_type.detach(), iona_e_angle.detach(), iona_e_coord.detach()]


class IONAVectorQuantizer(nn.Module):

    def __init__(self, iona_embedding_dim, iona_num_embeddings, iona_commitment_cost):
        super().__init__()
        self.iona_embedding_dim = iona_embedding_dim
        self.iona_num_embeddings = iona_num_embeddings
        self.iona_commitment_cost = iona_commitment_cost
        self.iona_embeddings = nn.Embedding(
            self.iona_num_embeddings, self.iona_embedding_dim
        )

    def forward(self, iona_x, iona_mask):
        iona_x = F.normalize(iona_x, p=2, dim=-1)
        iona_encoding_indices = self.iona_get_code_indices(iona_x)
        iona_quantized = self.iona_quantize(iona_encoding_indices)
        iona_q_latent_loss = F.mse_loss(
            iona_quantized[iona_mask], iona_x.detach()[iona_mask]
        )
        iona_e_latent_loss = F.mse_loss(
            iona_x[iona_mask], iona_quantized.detach()[iona_mask]
        )
        iona_loss = iona_q_latent_loss + self.iona_commitment_cost * iona_e_latent_loss
        iona_quantized = iona_x + (iona_quantized - iona_x).detach().contiguous()
        return (iona_quantized, iona_loss)

    def iona_get_code_indices(self, iona_x):
        iona_distances = (
            torch.sum(iona_x**2, dim=-1, keepdim=True)
            + torch.sum(
                F.normalize(self.iona_embeddings.weight, p=2, dim=-1).unsqueeze(0) ** 2,
                dim=-1,
            )
            - 2.0
            * torch.matmul(
                iona_x,
                F.normalize(self.iona_embeddings.weight.t(), p=2, dim=0).unsqueeze(0),
            )
        )
        iona_encoding_indices = torch.argmin(iona_distances, dim=-1)
        return iona_encoding_indices

    def iona_quantize(self, iona_encoding_indices):
        iona_N, iona_L = iona_encoding_indices.size()
        return F.normalize(
            self.iona_embeddings(iona_encoding_indices.reshape(-1)).reshape(
                iona_N, iona_L, self.iona_embedding_dim
            ),
            p=2,
            dim=-1,
        )
