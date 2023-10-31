
from multiprocessing import freeze_support
from pytorch_lightning.cli import ReduceLROnPlateau
#from apex import optimizers
from MolFormer.MolFormer import *
from MolFormer.tokenizer import MolTranBertTokenizer
import argparse
from MolFormer.DataMoudle import PropertyPredictionDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch import nn, optim, mean
from lightning_fabric.accelerators.cuda import is_cuda_available
from torchmetrics import AUROC,MeanAbsoluteError, MeanSquaredError
from pytorch_lightning import seed_everything
import os
import pandas as pd
import warnings
#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('highest')
class Head(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.nonlinear = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonlinear(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LightMolFormer(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super(LightMolFormer, self).__init__()
        self.config = config

        self.model = FineTuneMolFormer(tokenizer, config)
        self.tokenizer = tokenizer
        self.train_config = config
        # self.label_mean = config.label_mean
        # self.label_std = config.label_std
        # if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################
        if config.type_tasks == 'classify' and config.num_tasks == 1:
            self.loss = nn.CrossEntropyLoss()
        elif config.type_tasks == 'classify' and config.num_tasks > 1:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss = nn.L1Loss()
        if config.num_tasks > 1 and config.type_tasks == 'classify':
            self.train_auc = AUROC(task='multilabel',num_labels=config.num_tasks,average="macro",ignore_index=-1)
            self.test_auc = AUROC(task='multilabel',num_labels=config.num_tasks,average="macro",ignore_index=-1)
            self.val_auc = AUROC(task='multilabel',num_labels=config.num_tasks,average="macro",ignore_index=-1)
        elif config.num_tasks == 1 and config.type_tasks == 'classify':
            self.train_auc = AUROC(task='multiclass', num_classes=config.num_classes)
            self.test_auc = AUROC(task='multiclass', num_classes=config.num_classes)
            self.val_auc = AUROC(task='multiclass', num_classes=config.num_classes)
        elif config.type_tasks != 'classify':
            if config.dataset_name == 'qm9' or config.dataset_name == 'qm8':
                self.train_mase = MeanAbsoluteError()
                self.test_mase = MeanAbsoluteError()
                self.val_mase = MeanAbsoluteError()
            else:
                self.train_mase = MeanSquaredError(squared=False)
                self.test_mase = MeanSquaredError(squared=False)
                self.val_mase = MeanSquaredError(squared=False)
        # self.validation_outputs = {'loss': [], 'key_metric': [], 'key_metric2': [], 'avg_key_metric': []}
        # self.train_outputs = {'loss': [], 'key_metric': [], 'key_metric2': []}
        if self.config.num_tasks == 1:
            self.net = self.Net(
                config.n_embd, self.config.num_classes, dropout=config.dropout, type_tasks=self.config.type_tasks
            )
        else:
            self.net = self.Net(
                config.n_embd, self.config.num_tasks, dropout=config.dropout, type_tasks=self.config.type_tasks
            )

        self.save_hyperparameters(config)

    def add_adapter(self, adapter, **kwargs):

        self.model.add_adapter(adapter, **kwargs)
        self.model.load_from_pretrain()

    class Net(nn.Module):
        def __init__(self, smiles_embed_dim, num_tasks, dropout=0.1, type_tasks='classify'):
            super().__init__()
            self.desc_skip_connection = True
            self.type_tasks = type_tasks
            self.num_tasks = num_tasks
            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, num_tasks)  # classif

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)
            return z


    def configure_optimizers(self):
        #if self.config.model_type == 'prompt' or self.config.model_type == 'lora':
        if (self.config.model_type != 'none' and self.config.dataset_name != 'freesolv' and self.config.dataset_name !='lipo_nostd_2') or self.config.model_type == 'prompt' or self.config.model_type == 'lora':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
            #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
            if self.config.type_tasks == 'classify':
                mode = 'max'
            else:
                mode = 'min'
            lr_scheduler = ReduceLROnPlateau(optimizer, monitor='avg_val_metric',mode=mode, factor=0.1, patience=3, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": 'avg_val_metric'}
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                    #no_decay.add(fpn)
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    else:
                        decay.add(fpn)
            #no_decay.add('pos_emb')
            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            no_decay = no_decay-(decay & no_decay)
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            if self.config.type_tasks == 'regress':
                betas = (0.9, 0.999)
            else:
                betas = (0.9, 0.99)
            print('betas are {}'.format(betas))
            self.config.lr_multiplier = 1
            learning_rate = self.config.lr * self.config.lr_multiplier
            optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
            return optimizer

    def _calculate_loss(self, batch, mode="sum",train='train'):
        idx, mask, labels, labels_mask = batch
        token_embeddings = self.model(idx, mask)
        if token_embeddings.shape[1] != mask.shape[1]:
            mask = torch.cat((mask[:,:1],torch.ones((mask.shape[0],token_embeddings.shape[1]-mask.shape[1])).to(mask),mask[:,1:]),dim=1)
        if mode == 'sum':
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            token_embeddings = sum_embeddings / sum_mask
        elif mode == 'cls':
            token_embeddings = token_embeddings[:,0,:]
        elif mode == 'prompt_sum' and (self.config.model_type == 'prompt' or self.config.model_type == 'skip'):
            token_embeddings = token_embeddings[:, :self.config.num_tokens + 1, :]
            mask = mask[:,:self.config.num_tokens + 1]
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            token_embeddings = sum_embeddings / sum_mask
        if (self.config.dataset_name == 'freesolv'or self.config.dataset_name == 'esol') and self.config.std == True:
            labels = (labels-torch.Tensor(self.config.label_mean).to(labels))/torch.Tensor(self.config.label_std).to(labels)
        preds = self.net(token_embeddings)

        is_labeled = (~torch.isnan(labels)).to(torch.float32)
        labels_nan0 = torch.nan_to_num(labels)

        if self.config.num_tasks == 1 and self.config.num_classes != 1:
            if self.config.type_tasks == 'classify':
                loss = self.loss(preds, labels_nan0.long())
            else:
                loss = self.loss(preds, labels_nan0)
        else:
            if self.config.num_classes == 1:
                loss = self.loss(preds.squeeze(-1), labels_nan0.float())
            else:
                loss = (self.loss(preds, labels_nan0.float())* is_labeled).mean()
        if self.config.type_tasks == 'classify':
            labels = torch.nan_to_num(labels, -1)
            if self.config.num_tasks == 1:
                if train == 'train':
                    self.train_auc(torch.softmax(preds, dim=1), labels.long())
                    self.log('train_metric_step', self.train_auc)
                elif train == 'test':
                    self.test_auc(torch.softmax(preds, dim=1), labels.long())
                    self.log('test_metric_step', self.test_auc)
                else:
                    self.val_auc(torch.softmax(preds, dim=1), labels.long())
                    self.log('val_metric_step', self.val_auc)
            else:
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
            rounded_preds = preds.squeeze(-1).detach()
            # if (self.label_mean is not None) and (self.label_std is not None):
            #     rounded_preds = (rounded_preds*self.label_std.to(preds)+self.label_mean.to(preds))
            if train == 'train':
                self.train_mase(rounded_preds, labels)
                self.log('train_metric_step', self.train_mase)
            elif train == 'val':
                self.val_mase(rounded_preds, labels)
                self.log('val_metric_step', self.val_mase)
            else:
                self.test_mase(rounded_preds, labels)
                self.log('test_metric_step', self.test_mase)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, self.config.net_mode,train='train')
        self.log('train_loss', loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, self.config.net_mode,train='val')
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, self.config.net_mode,train='test')
        self.log('test_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        # 收集所有预测结果的loss和指标
        # 计算AUC值
        if self.config.type_tasks == 'classify':
            self.log('avg_val_metric', self.val_auc)
        else:
            self.log('avg_val_metric', self.val_mase)
        # 打印AUC值


    def on_train_epoch_end(self):
        # 计算并打印训练集上的MSE指标
        if self.config.type_tasks == 'classify':
            self.log('avg_train_metric', self.train_auc)
        else:
            self.log('avg_train_metric', self.train_mase)

    def on_test_epoch_end(self):
        # 计算并打印训练集上的MSE指标
        if self.config.type_tasks == 'classify':
            self.log('avg_test_metric', self.test_auc)
        else:
            self.log('avg_test_metric', self.test_mase)

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
        checkpoint['state_dict']= trainable_state_dict

        # 调用父类的方法保存检查点
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

#'checkpoints/bbbp/model/mutiheads/epoch=40-train_metric1=0.912-avg_val_metric=0.943.ckpt'
def main():
    MolFormer_config = {'n_head': 12, 'n_layer': 12, 'n_embd': 768, 'd_dropout': 0.1, 'num_feats': 32}
    MolFormer_config = argparse.Namespace(**MolFormer_config)
    train_config = {'data_root': 'Data/FineTuneData', 'dataset_name': 'lipo_nostd_2', 'measure_name': ['p_np'], 'aug':0,
                    'type_tasks': 'classify', 'num_tasks': 1, 'num_classes': 2, 'dropout': 0.1,'canonical':True,
                    'model_type': 'hfpcadapter','early_stop':True,'finetune_load_path':None,'std':False,
                    'train_batch_size':128,'val_batch_size':128, 'max_epochs': 200, 'num_workers': 24, 'lr': 3e-4,
                    'checkpoints_folder': 'checkpoints/MolFormer', 'seed': 2023, 'start_layer': 2,'net_mode':'sum','add':True,
                    'train_dataset_length': None, 'test_dataset_length': None, 'eval_dataset_length': None}
    lora_config = {'adapter_hidden_size':64,
                   'lora_r': 12, 'lora_alpha': 1, 'lora_dropout': 0.1,'enable': [True,False,True, False],
                   'num_tokens':64,'prompt_dropout':0.1,'project':215,'DEEP':True,'prompt_hidden_size':MolFormer_config.n_embd,
                   'sk_heads':2,'sk_hidden_size':64*2,'sk_dropout':0.1,
                   'muti_heads_hidden_size':1,'muti_heads':4,'head_factor':3,'head_dropout':0,
                   'muti_r':3,'muti_alpha':0.9,'muti_dropout':0,'muti_enable':[True,False,True,False],'muti_lora_heads':4,'muti_lora_factor':2}
    lora_config = argparse.Namespace(**lora_config)
    train_config = argparse.Namespace(**train_config)
    if train_config.dataset_name == 'tox21':
        train_config.measure_name = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                               'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        train_config.type_tasks = 'classify'
        train_config.num_tasks = len(train_config.measure_name)
    elif train_config.dataset_name == 'clintox':
        train_config.measure_name = ['FDA_APPROVED', 'CT_TOX']
        train_config.num_tasks = len(train_config.measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'muv':
        train_config.measure_name = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
        train_config.num_tasks = len(train_config.measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'sider':
        train_config.measure_name = [
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
        train_config.num_tasks = len(train_config.measure_name)
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'bbbp':
        train_config.measure_name = ['p_np']
        train_config.num_tasks = 1
        train_config.num_classes = 2
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'bace':
        train_config.measure_name = ['Class']
        train_config.num_tasks = 1
        train_config.num_classes = 2
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'qm9':
        train_config.measure_name = ['alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv']
        train_config.num_tasks = len(train_config.measure_name)
        train_config.type_tasks = 'regress'
    elif train_config.dataset_name == 'lipo':
        train_config.measure_name = ['y']
        train_config.num_tasks = 1
        train_config.num_classes = 1
        train_config.type_tasks = 'regress'
    elif train_config.dataset_name == 'lipo_nostd' or  train_config.dataset_name == 'lipo_nostd_2':
        train_config.measure_name = ['lipo']
        train_config.num_tasks = 1
        train_config.num_classes = 1
        train_config.type_tasks = 'regress'
    elif train_config.dataset_name == 'esol':
        train_config.measure_name = ['measured log solubility in mols per litre']
        train_config.num_tasks = 1
        train_config.num_classes = 1
        train_config.type_tasks = 'regress'
    elif train_config.dataset_name == 'freesolv':
        train_config.measure_name = ['expt']
        train_config.num_tasks = 1
        train_config.num_classes = 1
        train_config.type_tasks = 'regress'
    elif train_config.dataset_name == 'hiv':
        train_config.measure_name = ['HIV_active']
        train_config.num_tasks = 1
        train_config.num_classes = 2
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'estrogen':
        train_config.measure_name = ['alpha','beta']
        train_config.num_tasks = 2
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'toxcast':
        data = pd.read_csv('Data/FineTuneData/toxcast/valid.csv')
        train_config.measure_name = list(data.keys()[1:])
        train_config.num_tasks = len(train_config.measure_name)
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'
    elif train_config.dataset_name == 'metstab':
        train_config.measure_name = ['high','low']
        train_config.num_tasks = 2
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'

    checkpoint_root = os.path.join(train_config.checkpoints_folder, train_config.dataset_name)
    # checkpoint_model = os.path.join(checkpoint_root, 'model')
    # checkpoint_model = os.path.join(checkpoint_model, train_config.model_type)
    # checkpoint_log = os.path.join(checkpoint_root, 'log')
    assert type(train_config.measure_name) == type([])

    seed_everything(train_config.seed)
    tokenizer = MolTranBertTokenizer('MolFormer/bert_vocab.txt')
    if train_config.type_tasks == 'classify':
        mode = 'max'
    else:
        mode = 'min'

    if train_config.early_stop == True:
        early_stop_callback = EarlyStopping(monitor='train_loss', patience=20)
    else:
        early_stop_callback = EarlyStopping(monitor='train_loss', patience=train_config.max_epochs)

    train_config.label_mean = None
    train_config.label_std = None
    datamodule = PropertyPredictionDataModule(train_config, tokenizer)
    datamodule.prepare_data()
    if (train_config.dataset_name == 'freesolv' or train_config.dataset_name == 'esol') and train_config.std == True:
        train_config.label_mean = list(datamodule.train_ds.mean.numpy())
        train_config.label_std = list(datamodule.train_ds.std.numpy())
    # else:
    #     train_config.label_mean = None
    #     train_config.label_std = None

    config = argparse.Namespace(**vars(MolFormer_config), **vars(train_config), **vars(lora_config))
    datamodule = PropertyPredictionDataModule(config, tokenizer)
    datamodule.prepare_data()
    model = LightMolFormer(config, tokenizer)
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
                                                   mode=mode, filename='{epoch}-{avg_train_metric:.3f}-{avg_val_metric:.4f}')
    if config.finetune_load_path != None:
        finetune_state = torch.load(config.finetune_load_path)
        finetune_state = finetune_state["state_dict"]
        model.load_state_dict(finetune_state,strict=False)
    print(model)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=train_config.max_epochs,
        default_root_dir=checkpoint_root,
        devices=1 if is_cuda_available() else None,
        logger=logger,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule)


    data = torch.load(checkpoint_callback.best_model_path)
    config = argparse.Namespace(**data['hyper_parameters'])
    model = LightMolFormer(config,tokenizer)
    model.add_adapter(config.model_type, config=config)
    model.load_state_dict(data['state_dict'],strict=False)
    trainer.test(model,datamodule)

if __name__ == '__main__':
    freeze_support()
    main()
