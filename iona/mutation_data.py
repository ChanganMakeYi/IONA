import os
import copy
import math
import random
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index, index_to_one
from iona.primitives.transforms import iona_build_transforms
from iona.primitives.protein.parsers import iona_parse_biopython_structure


class IONAMutationDataset(Dataset):

    def __init__(
        self,
        iona_csv_path,
        iona_pdb_dir,
        iona_cache_dir,
        iona_num_cvfolds=3,
        iona_cvfold_index=0,
        iona_split="train",
        iona_patch_size=128,
        iona_reset=False,
    ):
        super().__init__()
        self.iona_csv_path = iona_csv_path
        self.iona_pdb_dir = iona_pdb_dir
        self.iona_cache_dir = iona_cache_dir
        os.makedirs(iona_cache_dir, exist_ok=True)
        self.iona_num_cvfolds = iona_num_cvfolds
        self.iona_cvfold_index = iona_cvfold_index
        self.iona_transform = iona_build_transforms(
            [
                {"type": "select_atom", "iona_resolution": "backbone+CB"},
                {
                    "type": "selected_region_fixed_size_patch",
                    "iona_select_attr": "mut_flag",
                    "iona_patch_size": iona_patch_size,
                },
            ]
        )
        self.iona_split = iona_split
        self.iona_entries_cache = os.path.join(iona_cache_dir, "entries.pkl")
        self.iona_entries = None
        self.iona_entries_full = None
        self.iona_load_entries(iona_reset)
        self.iona_structures_cache = os.path.join(iona_cache_dir, "structures.pkl")
        self.iona_structures = None
        self.iona_load_structures(iona_reset)
        iona_es_path = os.path.join(iona_cache_dir, "es_potential.pkl")
        self.iona_es_pot = (
            pickle.load(open(iona_es_path, "rb"))
            if os.path.exists(iona_es_path)
            else {}
        )
        if self.iona_es_pot:
            print(f"[es] 加载静电势 {len(self.iona_es_pot)} 结构")
        iona_valid = []
        for iona_e in self.iona_entries:
            _, iona_seq_map = self.iona_structures[iona_e["pdbcode"]]
            if any(
                (
                    (iona_m["chain"], iona_m["resseq"]) in iona_seq_map
                    for iona_m in iona_e["mutations"]
                )
            ):
                iona_valid.append(iona_e)
        if len(iona_valid) < len(self.iona_entries):
            print(
                f"[filter] 丢弃突变无法应用的条目: {len(self.iona_entries)} -> {len(iona_valid)}"
            )
        self.iona_entries = iona_valid

    def iona_load_entries(self, iona_reset):
        if not os.path.exists(self.iona_entries_cache) or iona_reset:
            self.iona_entries_full = self.iona_preprocess_entries()
        else:
            with open(self.iona_entries_cache, "rb") as iona_f:
                self.iona_entries_full = pickle.load(iona_f)
        iona_complex_to_entries = {}
        for iona_e in self.iona_entries_full:
            if iona_e["complex"] not in iona_complex_to_entries:
                iona_complex_to_entries[iona_e["complex"]] = []
            iona_complex_to_entries[iona_e["complex"]].append(iona_e)
        iona_complex_list = sorted(iona_complex_to_entries.keys())
        iona_explicit = None
        try:
            import pandas as _pd

            iona_df = _pd.read_csv(self.iona_csv_path, sep=";")
            iona_fc = f"fold_{self.iona_cvfold_index}"
            if iona_fc in iona_df.columns:
                iona_explicit = (
                    set(iona_df[iona_df[iona_fc] == "val"]["#Pdb"].astype(str)),
                    set(iona_df[iona_df[iona_fc] == "train"]["#Pdb"].astype(str)),
                )
        except Exception:
            iona_explicit = None
        if iona_explicit is not None:
            iona_val_c, iona_train_c = iona_explicit
            if self.iona_split == "val":
                iona_complexes_this = [
                    iona_c for iona_c in iona_complex_list if iona_c in iona_val_c
                ]
            elif self.iona_split == "all":
                iona_complexes_this = iona_complex_list
            else:
                iona_complexes_this = [
                    iona_c for iona_c in iona_complex_list if iona_c in iona_train_c
                ]
        else:
            iona_fs = os.environ.get("FOLD_SEED", "")
            iona_seed = int(iona_fs) if iona_fs.isdigit() else 3745754758
            random.Random(iona_seed).shuffle(iona_complex_list)
            iona_split_size = math.ceil(len(iona_complex_list) / self.iona_num_cvfolds)
            iona_complex_splits = [
                iona_complex_list[
                    iona_i * iona_split_size : (iona_i + 1) * iona_split_size
                ]
                for iona_i in range(self.iona_num_cvfolds)
            ]
            iona_val_split = iona_complex_splits.pop(self.iona_cvfold_index)
            iona_train_split = sum(iona_complex_splits, start=[])
            if self.iona_split == "val":
                iona_complexes_this = iona_val_split
            elif self.iona_split == "all":
                iona_complexes_this = iona_complex_list
            else:
                iona_complexes_this = iona_train_split
        iona_entries = []
        for iona_cplx in iona_complexes_this:
            iona_entries += iona_complex_to_entries[iona_cplx]
        self.iona_entries = iona_entries

    def iona_preprocess_entries(self):
        iona_entries = iona_read_mutation_entries(self.iona_csv_path, self.iona_pdb_dir)
        with open(self.iona_entries_cache, "wb") as iona_f:
            pickle.dump(iona_entries, iona_f)
        return iona_entries

    def iona_load_structures(self, iona_reset):
        if not os.path.exists(self.iona_structures_cache) or iona_reset:
            self.iona_structures = self.iona_preprocess_structures()
        else:
            with open(self.iona_structures_cache, "rb") as iona_f:
                self.iona_structures = pickle.load(iona_f)

    def iona_preprocess_structures(self):
        iona_structures = {}
        iona_pdbcodes = list(
            set([iona_e["pdbcode"] for iona_e in self.iona_entries_full])
        )
        for iona_pdbcode in tqdm(iona_pdbcodes, desc="Structures"):
            iona_parser = PDBParser(QUIET=True)
            iona_pdb_path = os.path.join(
                self.iona_pdb_dir, "{}.pdb".format(iona_pdbcode.upper())
            )
            iona_model = iona_parser.get_structure(None, iona_pdb_path)[0]
            iona_data, iona_seq_map = iona_parse_biopython_structure(iona_model)
            iona_structures[iona_pdbcode] = (iona_data, iona_seq_map)
        with open(self.iona_structures_cache, "wb") as iona_f:
            pickle.dump(iona_structures, iona_f)
        return iona_structures

    def __len__(self):
        return len(self.iona_entries)

    def __getitem__(self, iona_index):
        iona_entry = self.iona_entries[iona_index]
        iona_data, iona_seq_map = copy.deepcopy(
            self.iona_structures[iona_entry["pdbcode"]]
        )
        iona_data["ddG"] = iona_entry["ddG"]
        iona_data["complex"] = iona_entry["complex"]
        iona_data["num_muts"] = iona_entry["num_muts"]
        iona_data["entry_id"] = iona_entry["id"]
        iona_aa_mut = iona_data["aa"].clone()
        for iona_mut in iona_entry["mutations"]:
            iona_ch_rs_ic = (iona_mut["chain"], iona_mut["resseq"])
            if iona_ch_rs_ic not in iona_seq_map:
                continue
            iona_aa_mut[iona_seq_map[iona_ch_rs_ic]] = one_to_index(iona_mut["mt"])
        iona_data["aa_mut"] = iona_aa_mut
        iona_data["mut_flag"] = iona_data["aa"] != iona_data["aa_mut"]
        iona_L = iona_data["aa"].size(0)
        iona_pot = self.iona_es_pot.get(iona_entry["pdbcode"])
        if iona_pot is not None and len(iona_pot) == iona_L:
            iona_data["es_pot"] = torch.from_numpy(
                np.asarray(iona_pot, dtype=np.float32)
            )
            iona_data["es_mask"] = torch.ones(iona_L, dtype=torch.bool)
        else:
            iona_data["es_pot"] = torch.zeros(iona_L, dtype=torch.float32)
            iona_data["es_mask"] = torch.zeros(iona_L, dtype=torch.bool)
        if self.iona_transform is not None:
            iona_data = self.iona_transform(iona_data)
        return iona_data


def iona_read_mutation_entries(iona_csv_path, iona_pdb_dir, iona_block_list={"1KBH"}):
    iona_df = pd.read_csv(iona_csv_path, sep=";")
    iona_df["dG_wt"] = (
        8.314 / 4184 * (273.15 + 25.0) * np.log(iona_df["Affinity_wt_parsed"])
    )
    iona_df["dG_mut"] = (
        8.314 / 4184 * (273.15 + 25.0) * np.log(iona_df["Affinity_mut_parsed"])
    )
    iona_df["ddG"] = iona_df["dG_mut"] - iona_df["dG_wt"]

    def iona_parse_mut(iona_mut_name):
        return {
            "wt": iona_mut_name[0],
            "mt": iona_mut_name[-1],
            "chain": iona_mut_name[1],
            "resseq": int(iona_mut_name[2:-1]),
        }

    iona_entries = []
    for iona_i, iona_row in iona_df.iterrows():
        iona_pdbcode, _, _ = iona_row["#Pdb"].split("_")
        if iona_pdbcode in iona_block_list:
            continue
        if not os.path.exists(
            os.path.join(iona_pdb_dir, "{}.pdb".format(iona_pdbcode.upper()))
        ):
            continue
        if not np.isfinite(iona_row["ddG"]):
            continue
        iona_mutations = list(
            map(iona_parse_mut, iona_row["Mutation(s)_cleaned"].split(","))
        )
        iona_entry = {
            "id": iona_i,
            "complex": iona_row["#Pdb"],
            "pdbcode": iona_pdbcode,
            "mutations": iona_mutations,
            "num_muts": len(iona_mutations),
            "ddG": np.float32(iona_row["ddG"]),
        }
        iona_entries.append(iona_entry)
    return iona_entries


class IONASubstitutionDataset(Dataset):

    def __init__(self, iona_mutations):
        super().__init__()
        self.iona_pdb_path = "../data/7FAE_RBD_Fv.pdb"
        self.iona_data = None
        self.iona_seq_map = None
        self.iona_load_structure()
        self.iona_mutations = self.iona_parse_mutations(iona_mutations)
        self.iona_transform = iona_build_transforms(
            [
                {"type": "select_atom", "iona_resolution": "backbone+CB"},
                {
                    "type": "selected_region_fixed_size_patch",
                    "iona_select_attr": "mut_flag",
                    "iona_patch_size": 128,
                },
            ]
        )

    def iona_load_structure(self):
        if self.iona_pdb_path.endswith(".pdb"):
            iona_parser = PDBParser(QUIET=True)
        else:
            raise ValueError("Unknown file type.")
        iona_structure = iona_parser.get_structure(None, self.iona_pdb_path)
        self.iona_data, self.iona_seq_map = iona_parse_biopython_structure(
            iona_structure[0]
        )

    def iona_parse_mutations(self, iona_mutations):
        iona_parsed = []
        for iona_m in iona_mutations:
            iona_wt, iona_ch, iona_mt = (iona_m[0], iona_m[1], iona_m[-1])
            iona_seq = int(iona_m[2:-1])
            iona_pos = (iona_ch, iona_seq)
            if iona_pos not in self.iona_seq_map:
                continue
            if iona_mt == "*":
                for iona_mt_idx in range(20):
                    iona_mt = index_to_one(iona_mt_idx)
                    if iona_mt == iona_wt:
                        continue
                    iona_parsed.append(
                        {"position": iona_pos, "wt": iona_wt, "mt": iona_mt}
                    )
            else:
                iona_parsed.append({"position": iona_pos, "wt": iona_wt, "mt": iona_mt})
        return iona_parsed

    def __len__(self):
        return len(self.iona_mutations)

    def __getitem__(self, iona_index):
        iona_data = copy.deepcopy(self.iona_data)
        iona_mut = self.iona_mutations[iona_index]
        iona_mut_pos_idx = self.iona_seq_map[iona_mut["position"]]
        iona_data["mut_flag"] = torch.zeros(
            size=iona_data["aa"].shape, dtype=torch.bool
        )
        iona_data["mut_flag"][iona_mut_pos_idx] = True
        iona_data["aa_mut"] = iona_data["aa"].clone()
        iona_data["aa_mut"][iona_mut_pos_idx] = one_to_index(iona_mut["mt"])
        iona_data = self.iona_transform(iona_data)
        iona_data["ddG"] = 0
        iona_data["mutstr"] = "{}{}{}{}".format(
            iona_mut["wt"],
            iona_mut["position"][0],
            iona_mut["position"][1],
            iona_mut["mt"],
        )
        return iona_data
