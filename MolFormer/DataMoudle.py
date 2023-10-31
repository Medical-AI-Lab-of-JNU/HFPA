import os
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from MolFormer.tokenizer import MolTranBertTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from SmilesEnumerator import SmilesEnumerator


def normalize_smiles(smi, canonical, isomeric):
    try:
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveHs(mol)
        normalized = Chem.MolToSmiles(
            mol, canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def get_dataset(data_root, filename, dataset_len, aug, measure_name,canonical,mean=None,std=None):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df, measure_name, aug,canonical,mean,std)
    return dataset


class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name, aug=True,canonical=True,mean=None,std=None):
        df = df[['smiles'] + measure_name]
        self.measure_name = measure_name
        df = df.dropna(subset=['smiles'])
        if aug > 1:
            sme = SmilesEnumerator()
            for i in range(len(df)):
                label = df.loc[i, measure_name].values.tolist()
                smi = df.loc[i, 'smiles']
                for j in range(aug):
                    try:
                        aug_smile = sme.randomize_smiles(smi)
                        df.loc[len(df)] = [aug_smile]+label
                    except:
                        continue
        df = df.sample(frac=1)
        df['isomeric_smiles'] = df['smiles'].apply(lambda smi: normalize_smiles(smi, canonical=canonical, isomeric=False))
        df_good = df.dropna(subset=['isomeric_smiles'])  # TODO - Check why some rows are na

        len_new = len(df_good)
        # print('Dropped {} invalid smiles'.format(len(df) - len_new))
        print('aug_after_len:',len_new)
        self.df = df_good
        self.df = self.df.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.set_mean_and_std(mean,std)
    def __getitem__(self, index):
        mask = 1
        canonical_smiles = self.df.loc[index, 'isomeric_smiles']
        measures = self.df.loc[index, self.measure_name].values.tolist()

        if len(self.measure_name) > 1:
            mask = [0.0 if np.isnan(x) else 1.0 for x in measures]
            measures = [0.0 if np.isnan(x) else x for x in measures]
        else:
            mask = 0.0 if np.isnan(measures[0]) else 1.0
            measures = 0.0 if np.isnan(measures[0]) else measures[0]

        return canonical_smiles, measures, mask
        # idx,labels,label_mask

    def __len__(self):
        return len(self.df)

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            self.mean = torch.from_numpy(np.nanmean(self.df.loc[:, self.measure_name].values, axis=0))
        if std is None:
            self.std = torch.from_numpy(np.nanstd(self.df.loc[:, self.measure_name].values, axis=0))


class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(dataset_name, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "valid"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            train_filename,
            self.hparams.train_dataset_length,
            self.hparams.aug,
            measure_name=self.hparams.measure_name,
            canonical=self.hparams.canonical,
            mean=self.hparams.label_mean,
            std = self.hparams.label_std
        )

        val_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            valid_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
            canonical = self.hparams.canonical,
            mean=self.hparams.label_mean,
            std=self.hparams.label_std
        )

        test_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            test_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
            canonical=self.hparams.canonical,
            mean=self.hparams.label_mean,
            std=self.hparams.label_std
        )

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = [val_ds] + [test_ds]

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus([smile[0] for smile in batch], padding=True, add_special_tokens=True)
        return torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor(
            [smile[1] for smile in batch]), torch.tensor([smile[2] for smile in batch])

    #

    def val_dataloader(self):
        return DataLoader(
            ConcatDataset(self.val_ds),
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )