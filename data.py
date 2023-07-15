import csv
import pickle

import amd
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import random
import functools
import numpy as np
import pickle

from pdd_helpers import custom_PDD

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
        test_sampler = SequentialSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_set = torch.utils.data.Subset(dataset, indices[-test_size:])
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 # sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    batch_fea = []
    composition_fea = []
    cell_fea = []
    batch_target = []
    batch_cif_ids = []

    for i, (structure_features, comp_features, cell_features, target, cif_id) in enumerate(dataset_list):
        batch_fea.append(structure_features)
        composition_fea.append(comp_features)
        cell_fea.append(cell_features)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)

    return (pad_sequence(batch_fea, batch_first=True),
            pad_sequence(composition_fea, batch_first=True),
            torch.stack(cell_fea, dim=0)), \
           torch.stack(batch_target, dim=0), \
           batch_cif_ids


def collate_pretrain_pool(dataset_list):
    batch_fea = []
    composition_fea = []
    cell_fea = []
    batch_target = []
    batch_coords = []
    batch_cif_ids = []

    for i, (structure_features, comp_features, cell_features, target, coords, cif_id) in enumerate(dataset_list):
        batch_fea.append(structure_features)
        composition_fea.append(comp_features)
        cell_fea.append(cell_features)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        batch_coords.append(coords)

    return (pad_sequence(batch_fea, batch_first=True),
            pad_sequence(composition_fea, batch_first=True),
            torch.stack(cell_fea, dim=0)), \
           (torch.stack(batch_target, dim=0),
            pad_sequence(batch_coords, batch_first=True)), \
           batch_cif_ids


class LatticeEnergyData(Dataset):
    def __init__(self, filepath, k=60, collapse_tol=1e-4, constrained=False, seed=123, shuffle=True):
        periodic_sets = []
        if isinstance(filepath, list):
            for fp in filepath:
                r = amd.CifReader(fp)
                periodic_sets += [i for i in r]
        else:
            cached_data = "./data/" + os.path.basename(filepath) + "_raw_data"
            if os.path.exists(cached_data):
                with open(cached_data, "rb") as f:
                    periodic_sets = pickle.load(f)
            else:
                reader = amd.CifReader(filepath)
                periodic_sets = [i for j, i in enumerate(reader)]
                with open(cached_data, "wb") as f:
                    pickle.dump(periodic_sets, f)

        assert os.path.exists(filepath), 'CIF file does not exist!'
        self.k = k
        random.seed(seed)
        self.collapse_tol = float(collapse_tol)
        self.constrained = constrained
        if shuffle:
            random.shuffle(periodic_sets)
        self.ids = [ps.name for ps in periodic_sets]
        self.energies = [float(i.split("_")[0]) for i in self.ids]
        pdds = [amd.PDD(ps, k=self.k, collapse_tol=collapse_tol) for ps in periodic_sets]
        indices_to_keep = [i for i, pdd in enumerate(pdds) if pdd.shape[0] < 500]
        print("Keeping " + str(len(indices_to_keep)) + " / " + str(len(periodic_sets)) + " = " + str(
            len(indices_to_keep) / len(periodic_sets)))
        self.energies = [self.energies[i] for i in indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
        pdds = [pdds[i] for i in indices_to_keep]
        min_pdd = np.min(np.vstack([np.min(pdd, axis=0) for pdd in pdds]), axis=0)
        max_pdd = np.max(np.vstack([np.max(pdd, axis=0) for pdd in pdds]), axis=0)
        self.pdds = [np.hstack([pdd[:, 0, None], (pdd[:, 1:] - min_pdd[1:]) / (max_pdd[1:] - min_pdd[1:])]) for pdd in
                     pdds]

    def __len__(self):
        return len(self.pdds)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        return torch.Tensor(self.pdds[idx]), \
               torch.Tensor([float(self.energies[idx])]), \
               self.ids[idx]


class PDDData(Dataset):
    def __init__(self, filepath, k=60, collapse_tol=1e-4, composition=True, constrained=True, seed=123):
        self.filepath = filepath
        assert os.path.exists(filepath), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.filepath, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(seed)
        random.shuffle(self.id_prop_data)
        self.k = k
        random.seed(seed)
        self.collapse_tol = float(collapse_tol)
        self.constrained = constrained
        self.composition = composition

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        reader = amd.CifReader(os.path.join(self.filepath, cif_id + '.cif'))
        ps = reader.read()
        pdd, groups, inds, _ = custom_PDD(ps, k=self.k, collapse=True, collapse_tol=self.collapse_tol,
                                          constrained=self.constrained, lexsort=False)

        if self.composition:
            indices_in_graph = [i[0] for i in groups]
            atom_features = ps.types[indices_in_graph][:, None]
            pdd = np.hstack([pdd, atom_features])
            return torch.Tensor(pdd), \
                   torch.Tensor([float(target)]), \
                   cif_id
        else:
            pdd = amd.PDD(ps, k=self.k, collapse_tol=self.collapse_tol)
            return torch.Tensor(pdd), \
                   torch.Tensor([float(target)]), \
                   cif_id


class PDDDataNormalized(Dataset):
    def __init__(self, filepath, k=60, collapse_tol=1e-4, composition=True, constrained=True, seed=123):
        self.filepath = filepath
        assert os.path.exists(filepath), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.filepath, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(seed)
        random.shuffle(self.id_prop_data)
        k = int(k)
        self.k = k
        self.collapse_tol = float(collapse_tol)
        self.constrained = constrained
        self.composition = composition
        print("k: " + str(k))
        print("ct: " + str(self.collapse_tol))
        pdds = []
        periodic_sets = [amd.CifReader(os.path.join(filepath, cif[0] + ".cif")).read() for cif in self.id_prop_data]
        for ps in periodic_sets:
            pdd, groups, inds, _ = custom_PDD(ps, k=self.k, collapse=True, collapse_tol=self.collapse_tol,
                                              constrained=self.constrained, lexsort=False)
            indices_in_graph = [i[0] for i in groups]
            atom_features = ps.types[indices_in_graph][:, None]
            pdd = np.hstack([pdd, atom_features])
            pdds.append(pdd)

        min_pdd = np.min(np.vstack([np.min(pdd, axis=0) for pdd in pdds]), axis=0)
        max_pdd = np.max(np.vstack([np.max(pdd, axis=0) for pdd in pdds]), axis=0)
        self.pdds = [np.hstack(
            [pdd[:, 0, None], (pdd[:, 1:-1] - min_pdd[1:-1]) / (max_pdd[1:-1] - min_pdd[1:-1]), pdd[:, -1, None]]) for
            pdd
            in pdds]

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        return torch.Tensor(self.pdds[idx]), \
               torch.Tensor([float(target)]), \
               cif_id


class PDDDataPymatgen(Dataset):
    def __init__(self, structures, targets, k=60, collapse_tol=1e-4, composition=True, constrained=True, preprocess=True):
        k = int(k)
        self.k = k
        self.collapse_tol = float(collapse_tol)
        self.constrained = constrained
        self.composition = composition
        self.id_prop_data = targets
        pdds = []
        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in structures]
        self.cell_fea = [np.concatenate([np.sort(s.lattice.parameters[:3]), np.sort(s.lattice.parameters[3:])]) for s in
                         structures]
        if preprocess:
            i = 0
            for ps in periodic_sets:
                pdd, groups, inds, _ = custom_PDD(ps, k=self.k, collapse=True, collapse_tol=self.collapse_tol,
                                                  constrained=self.constrained, lexsort=False)
                indices_in_graph = [i[0] for i in groups]
                atom_features = ps.types[indices_in_graph][:, None]
                pdd = np.hstack([pdd, atom_features])
                pdds.append(pdd)
                i += 1

            min_pdd = np.min(np.vstack([np.min(pdd, axis=0) for pdd in pdds]), axis=0)
            max_pdd = np.max(np.vstack([np.max(pdd, axis=0) for pdd in pdds]), axis=0)
            self.pdds = [np.hstack(
                [pdd[:, 0, None], (pdd[:, 1:-1] - min_pdd[1:-1]) / (max_pdd[1:-1] - min_pdd[1:-1])]) for
                pdd
                in pdds]
            self.atom_fea = [pdd[:, -1, None] for pdd in pdds]
        else:
            self.neighbor_points = []
            self.atom_fea = []
            self.pdds = []
            for ps in periodic_sets:
                pdd, groups, inds, cloud = custom_PDD(ps, k=self.k, collapse=False, collapse_tol=self.collapse_tol,
                                                      constrained=self.constrained, lexsort=False)
                indices_in_graph = [i[0] for i in groups]
                self.pdds.append(pdd)
                og = cloud[indices_in_graph]
                n_points = cloud[inds]
                weighted_neighbors = np.hstack([og - n_points[:, i, :] for i in range(n_points.shape[1])])
                weighted_neighbors = np.hstack([pdd[:, 0].reshape((-1, 1)), weighted_neighbors])
                self.neighbor_points.append(weighted_neighbors)
                atom_features = ps.types[indices_in_graph][:, None]
                self.atom_fea.append(atom_features)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data.index[idx], self.id_prop_data.iloc[idx]
        return torch.Tensor(self.pdds[idx]), \
               torch.Tensor(self.atom_fea[idx]), \
               torch.Tensor(self.cell_fea[idx]), \
               torch.Tensor([float(target)]), \
               cif_id


class PretrainData(Dataset):
    def __init__(self, structures, k=12, collapse_tol=1e-4, composition=True, constrained=True):
        k = int(k)
        self.k = k
        self.collapse_tol = float(collapse_tol)
        self.constrained = constrained
        self.composition = composition
        self.id_prop_data = np.array(structures.index)
        pdds = []
        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in structures]
        i = 0
        self.m = []

        self.cell_fea = [np.concatenate([np.sort(s.lattice.parameters[:3]), np.sort(s.lattice.parameters[3:])]) for s in
                         structures]
        self.neighbor_points = []
        for ps in periodic_sets:
            pdd, groups, inds, cloud = custom_PDD(ps, k=self.k, collapse=False, collapse_tol=self.collapse_tol,
                                                  constrained=self.constrained, lexsort=False)
            indices_in_graph = [i[0] for i in groups]
            atom_features = ps.types[indices_in_graph][:, None]
            pdd = np.hstack([pdd, atom_features])
            pdds.append(pdd)
            i += 1
            self.m.append(pdd.shape[0])
            og = cloud[indices_in_graph]
            n_points = cloud[inds]
            self.neighbor_points.append(np.hstack([og - n_points[:, i, :] for i in range(n_points.shape[1])]))


        self.m = np.array(self.m)[:, None]
        min_pdd = np.min(np.vstack([np.min(pdd, axis=0) for pdd in pdds]), axis=0)
        max_pdd = np.max(np.vstack([np.max(pdd, axis=0) for pdd in pdds]), axis=0)
        self.pdds = [np.hstack(
            [pdd[:, 0, None], (pdd[:, 1:-1] - min_pdd[1:-1]) / (max_pdd[1:-1] - min_pdd[1:-1])]) for
            pdd
            in pdds]
        self.atom_fea = [pdd[:, -1, None] for pdd in pdds]

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx]
        # inputs -> pdds, atom fea, masked_indx
        # outputs -> masked_atom_fea, k-nn
        return torch.Tensor(self.pdds[idx]), \
               torch.Tensor(self.atom_fea[idx]), \
               torch.Tensor(self.cell_fea[idx]), \
               torch.LongTensor([self.m[idx]]), \
               torch.Tensor(self.neighbor_points[idx]), \
               cif_id
