import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from easydict import EasyDict
from .constants import (
    IONAAminoAcid,
    iona_max_num_heavyatoms,
    iona_restype_to_heavyatom_names,
    IONABackboneAtom,
)
from .icoord import iona_get_chi_angles, iona_get_backbone_torsions


def iona_get_residue_heavyatom_info(iona_res: Residue):
    iona_pos_heavyatom = torch.zeros([iona_max_num_heavyatoms, 3], dtype=torch.float)
    iona_mask_heavyatom = torch.zeros([iona_max_num_heavyatoms], dtype=torch.bool)
    iona_bfactor_heavyatom = torch.zeros([iona_max_num_heavyatoms], dtype=torch.float)
    iona_restype = IONAAminoAcid(iona_res.get_resname())
    for iona_idx, iona_atom_name in enumerate(
        iona_restype_to_heavyatom_names[iona_restype]
    ):
        if iona_atom_name == "":
            continue
        if iona_atom_name in iona_res:
            iona_pos_heavyatom[iona_idx] = torch.tensor(
                iona_res[iona_atom_name].get_coord().tolist(),
                dtype=iona_pos_heavyatom.dtype,
            )
            iona_mask_heavyatom[iona_idx] = True
            iona_bfactor_heavyatom[iona_idx] = iona_res[iona_atom_name].get_bfactor()
    return (iona_pos_heavyatom, iona_mask_heavyatom, iona_bfactor_heavyatom)


def iona_parse_pdb(iona_path, iona_model_id, iona_unknown_threshold=1.0):
    iona_parser = PDBParser()
    iona_structure = iona_parser.get_structure(None, iona_path)
    return iona_parse_biopython_structure(
        iona_structure[iona_model_id], iona_unknown_threshold=iona_unknown_threshold
    )


def iona_parse_mmcif_assembly(
    iona_path, iona_model_id, iona_assembly_id=0, iona_unknown_threshold=1.0
):
    iona_parser = MMCIFParser()
    iona_structure = iona_parser.get_structure(None, iona_path)
    iona_mmcif_dict = iona_parser._mmcif_dict
    if "_pdbx_struct_assembly_gen.asym_id_list" not in iona_mmcif_dict:
        return iona_parse_biopython_structure(
            iona_structure[iona_model_id], iona_unknown_threshold=iona_unknown_threshold
        )
    else:
        iona_assemblies = [
            tuple(iona_chains.split(","))
            for iona_chains in iona_mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"]
        ]
        iona_label_to_auth = {}
        for iona_label_asym_id, iona_auth_asym_id in zip(
            iona_mmcif_dict["_atom_site.label_asym_id"],
            iona_mmcif_dict["_atom_site.auth_asym_id"],
        ):
            iona_label_to_auth[iona_label_asym_id] = iona_auth_asym_id
        iona_model_real = list(
            {
                iona_structure[iona_model_id][iona_label_to_auth[iona_ch]]
                for iona_ch in iona_assemblies[iona_assembly_id]
            }
        )
        return iona_parse_biopython_structure(iona_model_real)


def iona_parse_biopython_structure(iona_entity, iona_unknown_threshold=1.0):
    iona_chains = Selection.unfold_entities(iona_entity, "C")
    iona_chains.sort(key=lambda iona_c: iona_c.get_id())
    iona_data = EasyDict(
        {
            "chain_id": [],
            "chain_nb": [],
            "resseq": [],
            "res_nb": [],
            "aa": [],
            "pos_heavyatom": [],
            "mask_heavyatom": [],
            "phi": [],
            "phi_mask": [],
            "psi": [],
            "psi_mask": [],
            "omega": [],
            "omega_mask": [],
            "chi": [],
            "chi_mask": [],
        }
    )
    iona_tensor_types = {
        "chain_nb": torch.LongTensor,
        "resseq": torch.LongTensor,
        "res_nb": torch.LongTensor,
        "aa": torch.LongTensor,
        "pos_heavyatom": torch.stack,
        "mask_heavyatom": torch.stack,
        "phi": torch.FloatTensor,
        "phi_mask": torch.BoolTensor,
        "psi": torch.FloatTensor,
        "psi_mask": torch.BoolTensor,
        "omega": torch.FloatTensor,
        "omega_mask": torch.BoolTensor,
        "chi": torch.stack,
        "chi_mask": torch.stack,
    }
    iona_count_aa, iona_count_unk = (0, 0)
    for iona_i, iona_chain in enumerate(iona_chains):
        iona_chain.atom_to_internal_coordinates()
        iona_seq_this = 0
        iona_residues = Selection.unfold_entities(iona_chain, "R")
        iona_residues.sort(
            key=lambda iona_res: (iona_res.get_id()[1], iona_res.get_id()[2])
        )
        for _, iona_res in enumerate(iona_residues):
            iona_resname = iona_res.get_resname()
            if not IONAAminoAcid.iona_is_aa(iona_resname):
                continue
            if not (
                iona_res.has_id("CA")
                and iona_res.has_id("C")
                and iona_res.has_id("N")
                and iona_res.has_id("O")
            ):
                continue
            iona_restype = IONAAminoAcid(iona_resname)
            iona_count_aa += 1
            if iona_restype == IONAAminoAcid.UNK:
                iona_count_unk += 1
                continue
            iona_data.chain_id.append(iona_chain.get_id())
            iona_data.chain_nb.append(iona_i)
            iona_data.aa.append(iona_restype)
            iona_pos_heavyatom, iona_mask_heavyatom, _ = (
                iona_get_residue_heavyatom_info(iona_res)
            )
            iona_data.pos_heavyatom.append(iona_pos_heavyatom)
            iona_data.mask_heavyatom.append(iona_mask_heavyatom)
            iona_phi, iona_psi, iona_omega = iona_get_backbone_torsions(iona_res)
            if iona_phi is None:
                iona_data.phi.append(0.0)
                iona_data.phi_mask.append(False)
            else:
                iona_data.phi.append(iona_phi)
                iona_data.phi_mask.append(True)
            if iona_psi is None:
                iona_data.psi.append(0.0)
                iona_data.psi_mask.append(False)
            else:
                iona_data.psi.append(iona_psi)
                iona_data.psi_mask.append(True)
            if iona_omega is None:
                iona_data.omega.append(0.0)
                iona_data.omega_mask.append(False)
            else:
                iona_data.omega.append(iona_omega)
                iona_data.omega_mask.append(True)
            iona_chi, _, iona_chi_mask, _ = iona_get_chi_angles(iona_restype, iona_res)
            iona_data.chi.append(iona_chi)
            iona_data.chi_mask.append(iona_chi_mask)
            iona_resseq_this = int(iona_res.get_id()[1])
            if iona_seq_this == 0:
                iona_seq_this = 1
            else:
                iona_d_CA_CA = torch.linalg.norm(
                    iona_data.pos_heavyatom[-2][IONABackboneAtom.CA]
                    - iona_data.pos_heavyatom[-1][IONABackboneAtom.CA],
                    ord=2,
                ).item()
                if iona_d_CA_CA <= 4.0:
                    iona_seq_this += 1
                else:
                    iona_d_resseq = iona_resseq_this - iona_data.resseq[-1]
                    iona_seq_this += max(2, iona_d_resseq)
            iona_data.resseq.append(iona_resseq_this)
            iona_data.res_nb.append(iona_seq_this)
    if len(iona_data.aa) == 0:
        return (None, None)
    if iona_count_unk / iona_count_aa >= iona_unknown_threshold:
        return (None, None)
    iona_seq_map = {}
    for iona_i, (iona_chain_id, iona_resseq) in enumerate(
        zip(iona_data.chain_id, iona_data.resseq)
    ):
        iona_seq_map[iona_chain_id, iona_resseq] = iona_i
    for iona_key, iona_convert_fn in iona_tensor_types.items():
        iona_data[iona_key] = iona_convert_fn(iona_data[iona_key])
    return (iona_data, iona_seq_map)
