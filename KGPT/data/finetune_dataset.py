from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from dgl.data.utils import load_graphs
import torch
import dgl.backend as F
import scipy.sparse as sps

from multiprocessing import Pool
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse

from KGPT.data.featurizer import smiles_to_graph_tune
from KGPT.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

SPLIT_TO_ID = {'train':0, 'val':1, 'test':2}


def preprocess_dataset(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.dataset}_{args.split}_{args.path_length}.pkl"
    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiless,
                  max_length=args.path_length,
                  n_virtual_nodes=2,
                  n_jobs = args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('saving graphs')
    save_graphs(cache_file_path, valid_graphs,
                labels={'labels': labels})

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')


    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    sp.save_npz(f"{args.data_path}/{args.dataset}/{args.split}_rdkfp1-7_512.npz", FP_sp_mat)
    np.savez_compressed(f"{args.data_path}/{args.dataset}/{args.split}_molecular_descriptors.npz", md=arr[:, 1:])


class MoleculeDataset(Dataset):
    def __init__(self, root_path, dataset, dataset_type, path_length=5, n_virtual_nodes=2, split_name=None, split=None):
        dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{split_name}_{path_length}.pkl")
        split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.npy")
        ecfp_path = os.path.join(root_path, f"{dataset}/{split_name}_rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/{split_name}_molecular_descriptors.npz")
        # Load Data
        df = pd.read_csv(dataset_path)
        if split is not None:
            use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split]]
        else: 
            use_idxs = np.arange(0, len(df))
        if not os.path.exists(ecfp_path) or not os.path.exists(md_path):
            args = {'data_path':root_path,'dataset':dataset,'path_length':path_length,'n_jobs':8,'split':split_name}
            preprocess_dataset(argparse.Namespace(**args))

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        self.df, self.fps, self.mds = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs]
        self.smiless = self.df['smiles'].tolist()
        self.use_idxs = use_idxs
        # Dataset Setting
        self.task_names = self.df.columns.drop(['smiles']).tolist()
        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        if dataset_type == 'classify':
            self._task_pos_weights = self.task_pos_weights()
        elif dataset_type == 'regress':
            self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds = self.fps,self.mds
    def __len__(self):
        return len(self.smiless)
    
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.labels[idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels,nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights
    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std


