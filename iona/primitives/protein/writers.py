import torch
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from .constants import IONAAminoAcid, iona_restype_to_heavyatom_names


def iona_save_pdb(iona_data, iona_path=None):

    def iona_mask_select(iona_v, iona_mask):
        if isinstance(iona_v, str):
            return "".join(
                [iona_s for iona_i, iona_s in enumerate(iona_v) if iona_mask[iona_i]]
            )
        elif isinstance(iona_v, list):
            return [iona_s for iona_i, iona_s in enumerate(iona_v) if iona_mask[iona_i]]
        elif isinstance(iona_v, torch.Tensor):
            return iona_v[iona_mask]
        else:
            return iona_v

    def iona_build_chain(
        iona_builder,
        iona_aa_ch,
        iona_pos_heavyatom_ch,
        iona_mask_heavyatom_ch,
        iona_chain_id_ch,
        iona_resseq_ch,
        iona_icode_ch,
    ):
        iona_builder.init_chain(iona_chain_id_ch[0])
        iona_builder.init_seg("    ")
        for (
            iona_aa_res,
            iona_pos_allatom_res,
            iona_mask_allatom_res,
            iona_resseq_res,
            iona_icode_res,
        ) in zip(
            iona_aa_ch,
            iona_pos_heavyatom_ch,
            iona_mask_heavyatom_ch,
            iona_resseq_ch,
            iona_icode_ch,
        ):
            iona_restype = IONAAminoAcid(iona_aa_res.item())
            iona_builder.init_residue(
                resname=str(iona_restype),
                field=" ",
                resseq=iona_resseq_res.item(),
                icode=iona_icode_res,
            )
            for iona_i, iona_atom_name in enumerate(
                iona_restype_to_heavyatom_names[iona_restype]
            ):
                if iona_atom_name == "":
                    continue
                if (~iona_mask_allatom_res[iona_i]).any():
                    continue
                if len(iona_atom_name) == 1:
                    iona_fullname = " %s  " % iona_atom_name
                elif len(iona_atom_name) == 2:
                    iona_fullname = " %s " % iona_atom_name
                elif len(iona_atom_name) == 3:
                    iona_fullname = " %s" % iona_atom_name
                else:
                    iona_fullname = iona_atom_name
                iona_builder.init_atom(
                    iona_atom_name,
                    iona_pos_allatom_res[iona_i].tolist(),
                    0.0,
                    1.0,
                    " ",
                    iona_fullname,
                )

    warnings.simplefilter("ignore", BiopythonWarning)
    iona_builder = StructureBuilder()
    iona_builder.init_structure(0)
    iona_builder.init_model(0)
    iona_unique_chain_nb = iona_data["chain_nb"].unique().tolist()
    for iona_ch_nb in iona_unique_chain_nb:
        iona_mask = iona_data["chain_nb"] == iona_ch_nb
        iona_aa = iona_mask_select(iona_data["aa"], iona_mask)
        iona_pos_heavyatom = iona_mask_select(iona_data["pos_heavyatom"], iona_mask)
        iona_mask_heavyatom = iona_mask_select(iona_data["mask_heavyatom"], iona_mask)
        iona_chain_id = iona_mask_select(iona_data["chain_id"], iona_mask)
        iona_resseq = iona_mask_select(iona_data["resseq"], iona_mask)
        iona_icode = iona_mask_select(iona_data["icode"], iona_mask)
        iona_build_chain(
            iona_builder,
            iona_aa,
            iona_pos_heavyatom,
            iona_mask_heavyatom,
            iona_chain_id,
            iona_resseq,
            iona_icode,
        )
    iona_structure = iona_builder.get_structure()
    if iona_path is not None:
        iona_io = PDBIO()
        iona_io.set_structure(iona_structure)
        iona_io.save(iona_path)
    return iona_structure
