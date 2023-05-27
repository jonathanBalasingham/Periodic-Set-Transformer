import amd
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import random
import functools

random.seed(0)


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    batch_fea = []
    batch_target = []
    batch_cif_ids = []

    for i, (features, target, cif_id) in enumerate(dataset_list):
        batch_fea.append(features)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)

    return pad_sequence(batch_fea, batch_first=True),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class LatticeEnergyData(Dataset):
    def __init__(self, filepath, k=6, collapse_tol=1e-4, constrained=False, seed=123, shuffle=True):
        periodic_sets = []
        if isinstance(filepath, list):
            for fp in filepath:
                r = amd.CifReader(fp)
                periodic_sets += [i for i in r]
        else:
            reader = amd.CifReader(filepath)
            periodic_sets = [i for j, i in enumerate(reader)]

        assert os.path.exists(filepath), 'CIF file does not exist!'
        self.k = k
        random.seed(seed)
        self.collapse_tol = float(collapse_tol)
        self.constrained = constrained
        if shuffle:
            random.shuffle(periodic_sets)
        self.ids = [ps.name for ps in periodic_sets]
        self.energies = [float(i.split("_")[0]) for i in self.ids]
        self.pdds = [amd.PDD(ps, k=self.k, collapse_tol=collapse_tol) for ps in periodic_sets]

    def __len__(self):
        return len(self.pdds)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        return torch.Tensor(self.pdds[idx]), \
            torch.Tensor([float(self.energies[idx])]), \
            self.ids[idx]
