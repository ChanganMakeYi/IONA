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
from common_utils.transforms import get_transform
from common_utils.protein.parsers import parse_biopython_structure


class SkempiDataset(Dataset):

    def __init__(self, csv_path, pdb_dir, cache_dir, num_cvfolds=3, cvfold_index=0, split='train', patch_size=128, reset=False):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.num_cvfolds = num_cvfolds
        self.cvfold_index = cvfold_index
        self.transform = get_transform([{'type': 'select_atom', 'resolution': 'backbone+CB'}, {'type': 'selected_region_fixed_size_patch', 'select_attr': 'mut_flag', 'patch_size': patch_size}])
        
        self.split = split

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self.structures = None
        self._load_structures(reset)

        # 方向B: 每残基 APBS 静电势(WT), 有则挂上供辅助任务
        es_path = os.path.join(cache_dir, 'es_potential.pkl')
        self.es_pot = pickle.load(open(es_path, 'rb')) if os.path.exists(es_path) else {}
        if self.es_pot: print(f'[es] 加载静电势 {len(self.es_pot)} 结构')

        # 过滤突变无法应用的条目(seq_map miss -> mut_flag 全 False -> patch 选区空 -> 崩)
        valid = []
        for e in self.entries:
            _, seq_map = self.structures[e['pdbcode']]
            if any((m['chain'], m['resseq']) in seq_map for m in e['mutations']):
                valid.append(e)
        if len(valid) < len(self.entries):
            print(f'[filter] 丢弃突变无法应用的条目: {len(self.entries)} -> {len(valid)}')
        self.entries = valid

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        # 若CSV含显式 fold_{k} 列(UniAIR标准划分), 优先用它; 否则用hash划分
        explicit = None
        try:
            import pandas as _pd
            _df = _pd.read_csv(self.csv_path, sep=';')
            _fc = f'fold_{self.cvfold_index}'
            if _fc in _df.columns:
                explicit = (set(_df[_df[_fc]=='val']['#Pdb'].astype(str)),
                            set(_df[_df[_fc]=='train']['#Pdb'].astype(str)))
        except Exception:
            explicit = None
        if explicit is not None:
            val_c, train_c = explicit
            if self.split == 'val':      complexes_this = [c for c in complex_list if c in val_c]
            elif self.split == 'all':    complexes_this = complex_list
            else:                        complexes_this = [c for c in complex_list if c in train_c]
        else:
            _fs = os.environ.get("FOLD_SEED", "")
            _seed = int(_fs) if _fs.isdigit() else 3745754758   # 默认死种子(可复现); 设FOLD_SEED环境变量换划分
            random.Random(_seed).shuffle(complex_list)
            split_size = math.ceil(len(complex_list) / self.num_cvfolds)
            complex_splits = [complex_list[i*split_size:(i+1)*split_size] for i in range(self.num_cvfolds)]
            val_split = complex_splits.pop(self.cvfold_index)
            train_split = sum(complex_splits, start=[])
            if self.split == 'val':      complexes_this = val_split
            elif self.split == 'all':    complexes_this = complex_list
            else:                        complexes_this = train_split

        entries = []
        for cplx in complexes_this:
            entries += complex_to_entries[cplx]
        self.entries = entries
        
    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.pdb_dir)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        structures = {}
        pdbcodes = list(set([e['pdbcode'] for e in self.entries_full]))
        for pdbcode in tqdm(pdbcodes, desc='Structures'):
            parser = PDBParser(QUIET=True)
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode.upper()))
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            structures[pdbcode] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        data['ddG'] = entry['ddG']
        data['complex'] = entry['complex']
        data['num_muts'] = entry['num_muts']
        data['entry_id'] = entry['id']   # 供ESM融合按id取序列特征

        aa_mut = data['aa'].clone()
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'])
            if ch_rs_ic not in seq_map: continue
            aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            
        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])

        # 挂每残基静电势 + 有效掩码(跟随 patch 一起被索引)
        L = data['aa'].size(0)
        pot = self.es_pot.get(entry['pdbcode'])
        if pot is not None and len(pot) == L:
            data['es_pot'] = torch.from_numpy(np.asarray(pot, dtype=np.float32))
            data['es_mask'] = torch.ones(L, dtype=torch.bool)
        else:
            data['es_pot'] = torch.zeros(L, dtype=torch.float32)
            data['es_mask'] = torch.zeros(L, dtype=torch.bool)

        if self.transform is not None:
            data = self.transform(data)

        return data
    

def load_skempi_entries(csv_path, pdb_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=';')

    df['dG_wt'] =  (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] =  (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mut'] - df['dG_wt']

    def _parse_mut(mut_name):
        return {'wt': mut_name[0], 'mt': mut_name[-1], 'chain': mut_name[1], 'resseq': int(mut_name[2:-1])}

    entries = []
    for i, row in df.iterrows():
        pdbcode, _, _ = row['#Pdb'].split('_')

        if pdbcode in block_list:
            continue
        if not os.path.exists(os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))):
            continue
        if not np.isfinite(row['ddG']):
            continue

        mutations = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        entry = {
            'id': i,
            'complex': row['#Pdb'],
            'pdbcode': pdbcode,
            'mutations': mutations,
            'num_muts': len(mutations),
            'ddG': np.float32(row['ddG']),
        }
        entries.append(entry)

    return entries


class SARSDataset(Dataset):

    def __init__(self, mutations):
        super().__init__()
        self.pdb_path = "../data/7FAE_RBD_Fv.pdb"

        self.data = None
        self.seq_map = None
        self._load_structure()

        self.mutations = self._parse_mutations(mutations)
        self.transform = get_transform([{'type': 'select_atom', 'resolution': 'backbone+CB'}, {'type': 'selected_region_fixed_size_patch', 'select_attr': 'mut_flag', 'patch_size': 128}])

    def _load_structure(self):
        if self.pdb_path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        else:
            raise ValueError('Unknown file type.')

        structure = parser.get_structure(None, self.pdb_path)
        self.data, self.seq_map = parse_biopython_structure(structure[0])

    def _parse_mutations(self, mutations):
        parsed = []

        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])
            pos = (ch, seq)
            if pos not in self.seq_map: continue

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'position': pos,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'position': pos,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data)
        mut = self.mutations[index]
        mut_pos_idx = self.seq_map[mut['position']]

        data['mut_flag'] = torch.zeros(size=data['aa'].shape, dtype=torch.bool)
        data['mut_flag'][mut_pos_idx] = True
        data['aa_mut'] = data['aa'].clone()
        data['aa_mut'][mut_pos_idx] = one_to_index(mut['mt'])
        data = self.transform(data)

        data['ddG'] = 0
        data['mutstr'] = '{}{}{}{}'.format(
            mut['wt'],
            mut['position'][0],
            mut['position'][1],
            mut['mt']
        )
        return data