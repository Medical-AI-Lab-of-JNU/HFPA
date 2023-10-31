import sys
from multiprocessing import freeze_support
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.cli import ReduceLROnPlateau
from KGPT.Kgpt import *
# from MolFormer_adapters import FineTune_ModelCheckpoint
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, ModelCheckpoint
from torch import nn, optim, mean
from lightning_fabric.accelerators.cuda import is_cuda_available
from torchmetrics import AUROC, MeanAbsoluteError, MeanSquaredError
from KGPT.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from KGPT.trainer import evaluator
from pytorch_lightning import seed_everything
import os
import warnings


# torch.autograd.set_detect_anomaly(True)
from KGPT.trainer.scheduler import PolynomialDecayLR


class FineTune_ModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, save_top_k, monitor='val_loss', mode='min', filename='model-{epoch:02d}-{val_loss:.2f}',
                 save_last=False):
        super().__init__(dirpath=dirpath, save_top_k=save_top_k, monitor=monitor, mode=mode, filename=filename,
                         save_last=save_last)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 获取非冻结参数的状态字典
        trainable_state_dict = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data

        # 更新检查点的状态字典
        checkpoint['state_dict'] = trainable_state_dict

        # 调用父类的方法保存检查点
        super().on_save_checkpoint(trainer, pl_module, checkpoint)


class LightKGPT(pl.LightningModule):
    def __init__(self, config):
        super(LightKGPT, self).__init__()
        self.config = config

        self.model = FineTunekgpt(config)
        self.label_mean = config.label_mean
        self.label_std = config.label_std
        self.train_last_batch = None
        self.val_last_batch = None
        self.test_last_batch = None
        if config.type_tasks == 'classify' and config.num_tasks == 1:
            self.loss = nn.CrossEntropyLoss()
        elif config.type_tasks == 'classify' and config.num_tasks > 1:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss = nn.MSELoss(reduction='none')
        if config.num_tasks > 1 and config.type_tasks == 'classify':
            if self.config.dataset == 'toxcast':
                self.evaluator = evaluator.Evaluator('toxcast', 'rocauc', config.num_tasks)
            else:
                self.train_auc = AUROC(task='multilabel',num_labels=config.num_tasks,average="macro",ignore_index=-1)

                self.val_auc = AUROC(task='multilabel',num_labels=config.num_tasks,average="macro",ignore_index=-1)

                self.test_auc = AUROC(task='multilabel', num_labels=config.num_tasks, average="macro", ignore_index=-1)
        elif config.num_tasks == 1 and config.type_tasks == 'classify':
            self.train_auc = AUROC(task='multiclass', num_classes=config.num_classes)
            self.test_auc = AUROC(task='multiclass', num_classes=config.num_classes)
            self.val_auc = AUROC(task='multiclass', num_classes=config.num_classes)
        elif config.type_tasks != 'classify':
            if config.dataset == 'qm9' or config.dataset == 'qm8':
                self.train_mase = MeanAbsoluteError()
                self.test_mase = MeanAbsoluteError()
                self.val_mase = MeanAbsoluteError()
            else:
                self.train_mase = MeanSquaredError(squared=False)
                self.test_mase = MeanSquaredError(squared=False)
                self.val_mase = MeanSquaredError(squared=False)
        if self.config.num_tasks == 1 and self.config.type_tasks == 'classify':
            self.model.model.predictor = get_predictor(d_input_feats=config.d_g_feats * 3, n_tasks=config.num_classes,
                                                       n_layers=2, predictor_drop=config.dropout, d_hidden_feats=256)
        else:
            self.model.model.predictor = get_predictor(d_input_feats=config.d_g_feats * 3, n_tasks=config.num_tasks,
                                                       n_layers=2, predictor_drop=config.dropout, d_hidden_feats=256)
        self.save_hyperparameters(config)
        del config

    def add_adapter(self, adapter, **kwargs):
        self.model.add_adapter(adapter, **kwargs)
        self.model.load_from_pretrain()

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        if self.config.type_tasks == 'classify':
            mode = 'max'
        else:
            mode = 'min'
        lr_scheduler = ReduceLROnPlateau(optimizer, monitor='avg_val_metric', mode=mode, factor=0.1, patience=3,
                                         verbose=True)

        # optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=self.config.max_epochs * self.config.train_dataset_len // 32 // 10,
        #                                  tot_updates=self.config.max_epochs * self.config.train_dataset_len // 32, lr=self.hparams.lr, end_lr=1e-9,
        #                                  power=1)
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": 'avg_val_metric'}


    def _calculate_loss(self, batch, train='train',last_batch = None):
        smiles, g, ecfp, md, labels = batch
        preds = self.model(g, ecfp, md)
        is_labeled = (~torch.isnan(labels)).to(torch.float32)
        labels_nan0 = torch.nan_to_num(labels)
        if (self.label_mean is not None) and (self.label_std is not None):
            labels_nan0 = (labels_nan0 - self.label_mean.to(labels)) / self.label_std.to(labels)
        if self.config.num_tasks == 1 and self.config.num_classes != 1:
            if self.config.type_tasks == 'classify':
                loss = self.loss(preds, labels_nan0.squeeze(1).long())
            else:
                loss = (self.loss(preds, labels_nan0) * is_labeled).mean()
        else:
            loss = (self.loss(preds, labels_nan0.float())* is_labeled).mean()
        if last_batch != None:
            last_preds,last_labels = last_batch
            preds = torch.cat((last_preds,preds),0)
            labels = torch.cat((last_labels,labels),0)

        if self.config.type_tasks == 'classify':
            if self.config.num_tasks == 1:
                labels = torch.nan_to_num(labels, -1)
                if train == 'train':
                    self.train_auc(torch.softmax(preds, dim=1), labels.squeeze(1).long())
                    self.log('train_metric_step', self.train_auc)
                elif train == 'test':
                    self.test_auc(torch.softmax(preds, dim=1), labels.squeeze(1).long())
                    self.log('test_metric_step', self.test_auc)
                else:
                    self.val_auc(torch.softmax(preds, dim=1), labels.squeeze(1).long())
                    self.log('val_metric_step', self.val_auc)
            else:
                #如果是toxcast，只在最后计算整个epoch的值
                if self.config.dataset == 'toxcast':
                    last_batch = (preds,labels)
                    return loss,last_batch
                # 沿着每一列求和以检查是否所有值都为零
                labels_nan0 = torch.nan_to_num(labels)
                column_sum = torch.sum(labels_nan0, dim=0)
                # 找到所有值都为零的列
                empty_columns =(column_sum == 0).nonzero().squeeze(1).tolist()
                if len(empty_columns) != 0:
                    last_batch = (preds,labels)
                    return loss,last_batch
                else:
                    last_batch = None
                labels = torch.nan_to_num(labels, -1)
                # 删除所有值都为零的列以及对应的预测列
                if train == 'train':
                    self.train_auc(torch.sigmoid(preds), (labels).long())
                    self.log('train_metric_step', self.train_auc)
                elif train == 'val':
                    self.val_auc(torch.sigmoid(preds), (labels).long())
                    self.log('val_metric_step', self.val_auc)
                else:

                    self.test_auc(torch.sigmoid(preds), (labels).long())
                    self.log('test_metric_step', self.test_auc)
        else:
            rounded_preds = preds.detach()
            if (self.label_mean is not None) and (self.label_std is not None):
                rounded_preds = (rounded_preds*self.label_std.to(preds)+self.label_mean.to(preds))
            if train == 'train':
                self.train_mase(rounded_preds, labels)
                self.log('train_metric_step', self.train_mase)
            elif train == 'val':
                self.val_mase(rounded_preds, labels)
                self.log('val_metric_step', self.val_mase)
            else:
                self.test_mase(rounded_preds, labels)
                self.log('test_metric_step', self.test_mase)
        return loss,last_batch

    def training_step(self, batch, batch_idx):
        loss,last_batch = self._calculate_loss(batch, train='train',last_batch=self.train_last_batch)
        if last_batch!=None:
            self.train_last_batch = last_batch
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss,last_batch = self._calculate_loss(batch, train='val',last_batch=self.val_last_batch)
        if last_batch!=None:
            self.val_last_batch = last_batch
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss,last_batch = self._calculate_loss(batch, train='test',last_batch=self.test_last_batch)
        if last_batch!=None:
            self.test_last_batch = last_batch
        self.log('test_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        # 收集所有预测结果的loss和指标
        # 计算AUC值
        if self.config.type_tasks == 'classify':
            if self.config.dataset == 'toxcast':
                preds,labels = self.val_last_batch
                val_auc = self.evaluator.eval(labels.detach().cpu(),preds.detach().cpu())
                self.log('avg_val_metric', val_auc)
                self.val_last_batch = None
            else:
                self.log('avg_val_metric', self.val_auc)
        else:
            self.log('avg_val_metric', self.val_mase)

    def on_train_epoch_end(self):
        # 计算并打印训练集上的MSE指标
        if self.config.type_tasks == 'classify':
            if self.config.dataset == 'toxcast':
                preds,labels = self.train_last_batch
                val_auc = self.evaluator.eval(labels.detach().cpu(),preds.detach().cpu())
                self.log('avg_train_metric', val_auc)
                self.train_last_batch = None
            else:
                self.log('avg_train_metric', self.train_auc)

        else:
            self.log('avg_train_metric', self.train_mase)

    def on_test_epoch_end(self):
        # 计算并打印训练集上的MSE指标
        if self.config.type_tasks == 'classify':
            if self.config.dataset == 'toxcast':
                preds,labels = self.test_last_batch
                val_auc = self.evaluator.eval(labels.detach().cpu(),preds.detach().cpu())
                self.log('avg_test_metric', val_auc)
                self.test_last_batch = None
            else:
                self.log('avg_test_metric', self.test_auc)
        else:
            self.log('avg_test_metric', self.test_mase)


# 'checkpoints/bbbp/model/mutiheads/epoch=40-train_metric1=0.912-avg_val_metric=0.943.ckpt'
def main():
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')
    kgpt_config = {'d_node_feats': 137, 'd_edge_feats': 14, 'd_g_feats': 768, 'd_hpath_ratio': 12, 'n_mol_layers': 12,
                   'path_length': 5, 'n_heads': 12,
                   'n_ffn_dense_layers': 2, 'input_drop': 0.0, 'attn_drop': 0.1, 'feat_drop': 0.1, 'GhT_dropout': 0,
                   'd_fps': None, 'd_mds': None, 'label_mean': None, 'label_std': None,
                   'candi_rate': 0.5, 'fp_disturb_rate': 0.5, 'md_disturb_rate': 0.5}
    train_config = {'data_path': 'Data/Kgpt_FineTuneData', 'dataset': 'freesolv', 'split': 'scaffold-0',
                    'type_tasks': 'classify', 'num_tasks': 1, 'num_classes': 2, 'dropout': 0.1,
                    'model_type': 'sk_cross', 'early_stop': True, 'finetune_load_path': None, 'vocab_size': None,
                    'train_batch_size': 128, 'val_batch_size': 128, 'max_epochs': 300, 'num_workers': 0, 'lr': 3e-4,
                    'checkpoints_folder': 'checkpoints/kgpt', 'seed': 2023, 'start_layer': 2, 'add': True}
    adapters_config = {'adapter_hidden_size': 64,
                       'lora_r': 3, 'lora_alpha': 1, 'lora_dropout': 0, 'enable': [True, False, True, False],
                       'num_tokens': 32, 'prompt_dropout': 0.1, 'project': 32, 'DEEP': True, 'prompt_hidden_size': 768,
                       'sk_heads': 2, 'sk_hidden_size': 32, 'sk_dropout': 0.1,
                       'muti_heads_hidden_size': 1, 'muti_heads': 4, 'head_factor': 3, 'head_dropout': 0,
                       'muti_r': 3, 'muti_alpha': 0.9, 'muti_dropout': 0, 'muti_enable': [True, False, True, False],
                       'muti_lora_heads': 4, 'muti_lora_factor': 2}
    kgpt_config = argparse.Namespace(**kgpt_config)
    adapters_config = argparse.Namespace(**adapters_config)
    train_config = argparse.Namespace(**train_config)
    if train_config.dataset == 'tox21':
        measure_name = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        train_config.num_tasks = len(measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'clintox':
        measure_name = ['FDA_APPROVED', 'CT_TOX']
        train_config.num_tasks = len(measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'muv':
        measure_name = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
        train_config.num_tasks = len(measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'sider':
        measure_name = [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders', 'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions',
            'Endocrine disorders', 'Surgical and medical procedures',
            'Vascular disorders', 'Blood and lymphatic system disorders',
            'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders', 'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
            'Renal and urinary disorders',
            'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders',
            'Nervous system disorders', 'Injury, poisoning and procedural complications'
        ]
        train_config.num_tasks = len(measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset== 'bbbp':
        measure_name = ['p_np']
        train_config.num_tasks = 1
        train_config.num_classes = 2
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'bace':
        measure_name = ['Class']
        train_config.num_tasks = 1
        train_config.num_classes = 2
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'qm9':
        measure_name = ['alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        train_config.num_tasks = len(measure_name)
        train_config.type_tasks = 'regress'
    elif train_config.dataset == 'lipo':
        measure_name = ['y']
        train_config.num_tasks = 1
        train_config.num_classes = 0
        train_config.type_tasks = 'regress'
    elif train_config.dataset == 'estrogen':
        measure_name = ['alpha','beta']
        train_config.num_tasks = 2
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'hiv':
        measure_name = ['HIV_active']
        train_config.num_tasks = 1
        train_config.num_classes = 2
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'esol':
        measure_name = ['logSolubility']
        train_config.num_tasks = 1
        train_config.num_classes = 1
        train_config.type_tasks = 'regress'
    elif train_config.dataset == 'freesolv':
        measure_name = ['expt']
        train_config.num_tasks = 1
        train_config.num_classes = 1
        train_config.type_tasks = 'regress'
    elif train_config.dataset == 'toxcast':
        data = pd.read_csv('Data/Kgpt_FineTuneData/toxcast/toxcast.csv')
        measure_name = list(data.keys()[1:])
        train_config.num_tasks = len(measure_name)
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'
    elif train_config.dataset == 'metstab':
        measure_name = ['high','low']
        train_config.num_tasks = len(measure_name)
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'

    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    train_config.vocab_size = vocab.vocab_size

    checkpoint_root = os.path.join(train_config.checkpoints_folder, train_config.dataset)
    #checkpoint_model = os.path.join(checkpoint_root, 'log')
    #checkpoint_model = os.path.join(checkpoint_model, train_config.model_type)
    #checkpoint_log = os.path.join(checkpoint_root, 'log')

    seed_everything(train_config.seed)

    if train_config.type_tasks == 'classify':
        mode = 'max'
    else:
        mode = 'min'
    config = argparse.Namespace(**vars(kgpt_config), **vars(train_config), **vars(adapters_config))
    datamodule = PropertyPredictionDataModule(config)
    datamodule.prepare_data()
    config.d_fps = datamodule.d_fps
    config.d_mds = datamodule.d_mds
    config.label_mean = datamodule.label_mean if datamodule.label_mean is not None else None
    config.label_std = datamodule.label_std if datamodule.label_std is not None else None
    config.train_dataset_len = len(datamodule.train_ds)
    model = LightKGPT(config)
    model.add_adapter(train_config.model_type, config=config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        # version=run_name,
        name=train_config.model_type,
        default_hp_metric=False,
    )
    checkpoint_callback = FineTune_ModelCheckpoint(dirpath=logger.log_dir, monitor='avg_val_metric', save_top_k=1,
                                                   mode=mode,
                                                   filename='{epoch}-{avg_train_metric:.3f}-{avg_val_metric:.4f}')
    if train_config.early_stop == True:
        early_stop_callback = EarlyStopping(monitor='train_loss', patience=20)
    else:
        early_stop_callback = EarlyStopping(monitor='train_loss', patience=train_config.max_epochs)

    if config.finetune_load_path != None:
        finetune_state = torch.load(config.finetune_load_path)
        finetune_state = finetune_state["state_dict"]
        model.load_state_dict(finetune_state, strict=False)
    print(model)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=train_config.max_epochs,
        default_root_dir=checkpoint_root,
        devices=1 if is_cuda_available() else None,
        # accelerator="cpu",
        logger=logger,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule)

    # model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path,strict=False)
    data = torch.load(checkpoint_callback.best_model_path)
    config = argparse.Namespace(**data['hyper_parameters'])
    model = LightKGPT(config)
    model.add_adapter(config.model_type, config=config)
    model.load_state_dict(data['state_dict'], strict=False)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    freeze_support()
    main()
