import torch
import numpy as np
from Bio.PDB.Residue import Residue
from .constants import IONAAminoAcid, iona_chi_angles_atoms, iona_chi_pi_periodic


def iona_get_chi_angles(iona_restype: IONAAminoAcid, iona_res: Residue):
    iona_ic = iona_res.internal_coord
    iona_chi_angles = torch.zeros([4])
    iona_chi_angles_alt = torch.zeros([4])
    iona_chi_angles_mask = torch.zeros([4], dtype=torch.bool)
    iona_count_chi_angles = len(iona_chi_angles_atoms[iona_restype])
    if iona_ic is not None:
        for iona_i in range(iona_count_chi_angles):
            iona_angle_name = "chi%d" % (iona_i + 1)
            if iona_ic.get_angle(iona_angle_name) is not None:
                iona_angle = np.deg2rad(iona_ic.get_angle(iona_angle_name))
                iona_chi_angles[iona_i] = iona_angle
                iona_chi_angles_mask[iona_i] = True
                if iona_chi_pi_periodic[iona_restype][iona_i]:
                    if iona_angle >= 0:
                        iona_angle_alt = iona_angle - np.pi
                    else:
                        iona_angle_alt = iona_angle + np.pi
                    iona_chi_angles_alt[iona_i] = iona_angle_alt
                else:
                    iona_chi_angles_alt[iona_i] = iona_angle
    iona_chi_complete = iona_count_chi_angles == iona_chi_angles_mask.sum().item()
    return (
        iona_chi_angles,
        iona_chi_angles_alt,
        iona_chi_angles_mask,
        iona_chi_complete,
    )


def iona_get_backbone_torsions(iona_res: Residue):
    iona_ic = iona_res.internal_coord
    if iona_ic is None:
        return (None, None, None)
    iona_phi, iona_psi, iona_omega = (
        iona_ic.get_angle("phi"),
        iona_ic.get_angle("psi"),
        iona_ic.get_angle("omega"),
    )
    if iona_phi is not None:
        iona_phi = np.deg2rad(iona_phi)
    if iona_psi is not None:
        iona_psi = np.deg2rad(iona_psi)
    if iona_omega is not None:
        iona_omega = np.deg2rad(iona_omega)
    return (iona_phi, iona_psi, iona_omega)
