
from multiprocessing import freeze_support

import torchmetrics
from pytorch_lightning.cli import ReduceLROnPlateau
from apex import optimizers
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
        if config.type_tasks == 'classify' and config.num_tasks == 1:
            self.loss = nn.CrossEntropyLoss()
        elif config.type_tasks == 'classify' and config.num_tasks > 1:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.L1Loss()

        self.main_metrice = config.main_metric
        self.metrics = {key: [] for key in config.metrics}
        self.configure_metrics(config.metrics, config.num_classes)
        if self.config.num_tasks == 1:
            self.net = self.Net(
                config.n_embd, self.config.num_classes, dropout=config.dropout, type_tasks=self.config.type_tasks
            )
        else:
            self.net = self.Net(
                config.n_embd, self.config.num_tasks, dropout=config.dropout, type_tasks=self.config.type_tasks
            )

        self.save_hyperparameters(config)

    def configure_metrics(self, metrics, num_classes):
        # 设置所需要的指标
        for key in metrics:
            for i in range(3):
                if self.config.type_tasks == 'classify':
                    if num_classes > 2:
                        self.metrics[key].append(
                            getattr(torchmetrics, key)(task='multilabel',num_labels=self.config.num_tasks,average="macro",ignore_index=-1).to(self.device))
                    else:
                        self.metrics[key].append(
                            getattr(torchmetrics, key)('binary').to(self.device))
                elif self.config.type_tasks != 'classify':
                    if self.config.dataset_name == 'qm9' or self.config.dataset_name == 'qm8':
                        self.metrics[key].append(
                            getattr(torchmetrics, key)().to(self.device))
                    else:
                        if key == 'MeanSquaredError':
                            self.metrics[key].append(
                                getattr(torchmetrics, key)(squared = False).to(self.device))
                        else:
                            self.metrics[key].append(
                                getattr(torchmetrics, key)(task="multiclass", num_classes=num_classes).to(self.device))

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
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": f'val_{self.main_metrice}_epoch'}
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
                loss = self.loss((preds*is_labeled).squeeze(), labels_nan0.float())
        if self.config.type_tasks == 'classify':
            labels = torch.nan_to_num(labels, -1)
            if self.config.num_tasks == 1:
                self.log_metrics_step(torch.softmax(preds, dim=1)[:,1],labels.long(),train)
            else:
                self.log_metrics_step(torch.sigmoid(preds), (labels).long(),train)
        else:
            rounded_preds = preds.squeeze(-1).detach()
            self.log_metrics_step(rounded_preds,labels,train)
        return loss

    def log_metrics_step(self, preds, labels, mode='train'):
        index2mode = ['train', 'val', 'test']
        metrics = list(self.metrics.keys())
        index = index2mode.index(mode)
        # 记录训练每一步的指标{训练状态}——{指标名字}——{step}
        for metrice in range(len(self.metrics)):
            self.metrics[metrics[metrice]][index].to(self.device)
            # 只显示训练集的每一步主要指标
            #value = self.metrics[metrics[metrice]][index].update(preds, labels)
            if metrics[metrice] == self.main_metrice and index == 0:
                value = self.metrics[metrics[metrice]][index](preds, labels)
                self.log(f'{index2mode[index]}_{metrics[metrice]}_step',
                         value, prog_bar=True, on_step=True,
                         on_epoch=False)
            else:
                # if index == 1 and metrice == 0 :
                #     self.pred[len(self.pred)-1].append(preds.cpu().numpy())
                #     self.label[len(self.label)-1].append(labels.cpu().numpy())
                self.metrics[metrics[metrice]][index].update(preds, labels)

    def log_metrics_epoch(self, mode='train'):
        index2mode = ['train', 'val', 'test']
        metrics = list(self.metrics.keys())
        index = index2mode.index(mode)

        # 记录训练每一个epoch的指标{训练状态}——{指标名字}——{epoch}
        for metrice in range(len(self.metrics)):
            # 只显示验证集的epoch主要指标
            value = self.metrics[metrics[metrice]][index].compute()
            if metrics[metrice] == self.main_metrice and index == 1:
                self.log(f'{index2mode[index]}_{metrics[metrice]}_epoch',
                         value, prog_bar=True)
            else:
                self.log(f'{index2mode[index]}_{metrics[metrice]}_epoch', value)
            self.metrics[metrics[metrice]][index].reset()


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
        self.log_metrics_epoch(mode='val')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse configuration for model training.')

    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--d_dropout', type=float, default=0.1, help='Dropout rate')

    parser.add_argument('--data_root', type=str, default='Data/FineTuneData', help='Data root directory')
    parser.add_argument('--dataset_name', type=str, default='bbbp', help='Dataset name')
    parser.add_argument('--measure_name', nargs='+', default=['p_np'], help='Measure names')
    parser.add_argument('--aug', type=int, default=0, help='Data augmentation flag')
    parser.add_argument('--type_tasks', type=str, default='classify', help='Type of tasks')
    parser.add_argument('--num_tasks', type=int, default=1, help='Number of tasks')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for the model')
    parser.add_argument('--canonical', action='store_true', help='Use canonical smiles')
    parser.add_argument('--model_type', type=str, default='sk', help='Model type')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--finetune_load_path', type=str, help='Path to load finetuned model')
    parser.add_argument('--std', action='store_false', help='Standardize data')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of training epochs')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints/MolFormer', help='Checkpoints folder')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducibility')
    parser.add_argument('--start_layer', type=int, default=2, help='Start layer for fine-tuning')
    parser.add_argument('--net_mode', type=str, default='sum', help='Network mode for combining features')
    parser.add_argument('--add', action='store_true', help='Additional configuration flag')

    parser.add_argument('--adapter_hidden_size', type=int, default=64, help='Hidden size of the adapter')
    parser.add_argument('--heads', type=int, default=64, help='Head num of the HFPA')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of the HFPA')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate of the HFPA')

    train_config = parser.parse_args()
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
    elif train_config.dataset_name == 'toxcast':
        data = pd.read_csv('Data/FineTuneData/toxcast/valid.csv')
        train_config.measure_name = list(data.keys()[1:])
        train_config.num_tasks = len(train_config.measure_name)
        train_config.num_classes = 1
        train_config.type_tasks = 'classify'

    checkpoint_root = os.path.join(train_config.checkpoints_folder, train_config.dataset_name)
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
    config = train_config
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
