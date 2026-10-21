import math
import functools
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from iona.mutation_data import IONAMutationDataset, IONASubstitutionDataset

iona_DEFAULT_PAD_VALUES = {"aa": 21, "chain_nb": -1, "chain_id": " "}


class IONAPaddedBatch(object):

    def __init__(
        self, iona_length_ref_key="aa", iona_pad_values=iona_DEFAULT_PAD_VALUES
    ):
        super().__init__()
        self.iona_length_ref_key = iona_length_ref_key
        self.iona_pad_values = iona_pad_values

    @staticmethod
    def iona_pad_last(iona_x, iona_n, iona_value=0):
        if isinstance(iona_x, torch.Tensor):
            assert iona_x.size(0) <= iona_n
            if iona_x.size(0) == iona_n:
                return iona_x
            iona_pad_size = [iona_n - iona_x.size(0)] + list(iona_x.shape[1:])
            iona_pad = torch.full(iona_pad_size, fill_value=iona_value).to(iona_x)
            return torch.cat([iona_x, iona_pad], dim=0)
        elif isinstance(iona_x, list):
            iona_pad = [iona_value] * (iona_n - len(iona_x))
            return iona_x + iona_pad
        else:
            return iona_x

    @staticmethod
    def iona_get_common_keys(iona_list_of_dict):
        iona_keys = set(iona_list_of_dict[0].keys())
        for iona_d in iona_list_of_dict[1:]:
            iona_keys = iona_keys.intersection(iona_d.keys())
        return iona_keys

    def iona_get_pad_value(self, iona_key):
        if iona_key not in self.iona_pad_values:
            return 0
        return self.iona_pad_values[iona_key]

    def __call__(self, iona_data_list):
        iona_max_length = max(
            [
                iona_data[self.iona_length_ref_key].size(0)
                for iona_data in iona_data_list
            ]
        )
        iona_max_length = math.ceil(iona_max_length / 8) * 8
        iona_keys = self.iona_get_common_keys(iona_data_list)
        iona_data_list_padded = []
        for iona_data in iona_data_list:
            iona_data_padded = {
                iona_k: self.iona_pad_last(
                    iona_v, iona_max_length, iona_value=self.iona_get_pad_value(iona_k)
                )
                for iona_k, iona_v in iona_data.items()
                if iona_k in iona_keys
            }
            iona_data_list_padded.append(iona_data_padded)
        iona_batch = default_collate(iona_data_list_padded)
        return iona_batch


def iona_inf_iterator(iona_iterable):
    iona_iterator = iona_iterable.__iter__()
    while True:
        try:
            yield iona_iterator.__next__()
        except StopIteration:
            iona_iterator = iona_iterable.__iter__()


class IONADataManager(object):

    def __init__(self, iona_config, iona_num_cvfolds, iona_num_workers=4):
        super().__init__()
        self.iona_config = iona_config
        self.iona_num_cvfolds = iona_num_cvfolds
        self.iona_num_workers = iona_num_workers
        self.iona_train_loaders = []
        self.iona_val_loaders = []
        for iona_fold in range(iona_num_cvfolds):
            iona_train_loader, iona_val_loader, iona_pretrain_loader = (
                self.iona_init_loaders(iona_fold)
            )
            self.iona_train_loaders.append(iona_train_loader)
            self.iona_val_loaders.append(iona_val_loader)
        self.iona_pretrain_loader = iona_pretrain_loader

    def iona_init_loaders(self, iona_fold):
        iona_dataset = functools.partial(
            IONAMutationDataset,
            iona_csv_path=getattr(
                self.iona_config, "iona_csv_path", "../data/SKEMPI_v2/skempi_v2.csv"
            ),
            iona_pdb_dir=getattr(
                self.iona_config, "iona_pdb_dir", "../data/SKEMPI_v2/PDBs"
            ),
            iona_cache_dir=getattr(
                self.iona_config, "iona_cache_dir", "../data/SKEMPI_v2_cache"
            ),
            iona_num_cvfolds=self.iona_num_cvfolds,
            iona_cvfold_index=iona_fold,
            iona_patch_size=self.iona_config.iona_patch_size,
        )
        iona_train_dataset = iona_dataset(iona_split="train")
        iona_val_dataset = iona_dataset(iona_split="val")
        iona_pretrain_dataset = iona_dataset(iona_split="all")
        iona_train_cplx = set(
            [iona_e["complex"] for iona_e in iona_train_dataset.iona_entries]
        )
        iona_val_cplx = set(
            [iona_e["complex"] for iona_e in iona_val_dataset.iona_entries]
        )
        iona_leakage = iona_train_cplx.intersection(iona_val_cplx)
        assert len(iona_leakage) == 0, f"data leakage {iona_leakage}"
        iona_train_loader = DataLoader(
            iona_train_dataset,
            batch_size=self.iona_config.iona_batch_size,
            shuffle=True,
            collate_fn=IONAPaddedBatch(),
            num_workers=self.iona_num_workers,
        )
        iona_train_iterator = iona_inf_iterator(iona_train_loader)
        iona_val_loader = DataLoader(
            iona_val_dataset,
            batch_size=self.iona_config.iona_batch_size,
            shuffle=False,
            collate_fn=IONAPaddedBatch(),
            num_workers=self.iona_num_workers,
        )
        iona_pretrain_loader = DataLoader(
            iona_pretrain_dataset,
            batch_size=self.iona_config.iona_batch_size,
            shuffle=True,
            collate_fn=IONAPaddedBatch(),
            num_workers=self.iona_num_workers,
        )
        iona_pretrain_iterator = iona_inf_iterator(iona_pretrain_loader)
        print(
            "Fold %d: Train %d, Val %d, All %d"
            % (
                iona_fold,
                len(iona_train_dataset),
                len(iona_val_dataset),
                len(iona_pretrain_dataset),
            )
        )
        return (iona_train_iterator, iona_val_loader, iona_pretrain_iterator)

    def iona_get_train_loader(self, iona_fold):
        return self.iona_train_loaders[iona_fold]

    def iona_get_val_loader(self, iona_fold):
        return self.iona_val_loaders[iona_fold]


def iona_get_SARS_data(iona_batch_size, iona_config):
    iona_dataset = IONASubstitutionDataset(iona_mutations=iona_config.iona_mutations)
    iona_loader = DataLoader(
        iona_dataset,
        batch_size=iona_batch_size,
        shuffle=False,
        collate_fn=IONAPaddedBatch(),
    )
    return iona_loader
