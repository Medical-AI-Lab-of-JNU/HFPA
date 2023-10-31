from argparse import Namespace

from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from KGPT.data.collator import Collator_tune
from KGPT.data.finetune_dataset import MoleculeDataset
from adapters.adapters import *
from KGPT.model.light import LiGhTPredictor as LiGhT, init_params
from KGPT.model.light import Residual, MLP,TripletTransformer
import pytorch_lightning as pl
from dgl import function as fn
from dgl.nn.functional import edge_softmax

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor


class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.dataset_name = hparams.dataset
        self.args = hparams
        # self.g = torch.Generator()
        # self.g.manual_seed(self.args.seed)
        self.collator = Collator_tune(hparams.path_length)

    def prepare_data(self) -> None:
        train_dataset = MoleculeDataset(root_path=self.args.data_path, dataset=self.args.dataset,
                                        dataset_type=self.args.type_tasks,
                                        split_name=f'{self.args.split}', split='train')
        val_dataset = MoleculeDataset(root_path=self.args.data_path, dataset=self.args.dataset,
                                      dataset_type=self.args.type_tasks,
                                      split_name=f'{self.args.split}', split='val')
        test_dataset = MoleculeDataset(root_path=self.args.data_path, dataset=self.args.dataset,
                                       dataset_type=self.args.type_tasks,
                                       split_name=f'{self.args.split}', split='test')

        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset
        self.d_fps = train_dataset.d_fps
        self.d_mds = train_dataset.d_mds
        self.label_mean = train_dataset.mean
        self.label_std = train_dataset.std

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers,
                                  drop_last=True, collate_fn=self.collator)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers,
                                 drop_last=False, collate_fn=self.collator)
        return test_loader

    def val_dataloader(self):
        val_loader = DataLoader(ConcatDataset([self.val_ds,self.test_ds]), batch_size=self.args.val_batch_size, shuffle=False,
                                num_workers=self.args.num_workers,
                                drop_last=False, collate_fn=self.collator)
        return val_loader


class sk_Residual(Residual):
    def __init__(self, config, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(sk_Residual, self).__init__(d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation)
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats * 4)
        self.feat_dropout = nn.Dropout(feat_drop)
        self.norm2 = nn.LayerNorm(d_out_feats)
        self.adapter_heads = config.sk_heads  # heads的数目
        self.adapter_heads_size = int(config.sk_hidden_size / self.adapter_heads)
        self.sk = sk_adapter(d_out_feats, self.adapter_heads, self.adapter_heads_size, dropout=config.sk_dropout)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        x = self.sk(x)
        x = self.norm2(x)
        return x

    def make_finetune_trainable(self):
        self.sk.requires_grad_(True)
        self.norm2.requires_grad_(True)
        self.norm.requires_grad_(False)
        self.in_proj.requires_grad_(False)
        self.ffn.requires_grad_(False)
        self.feat_dropout.requires_grad_(False)


class sk_TripletTransformer(TripletTransformer):
    def __init__(self,config,d_feats,
                d_hpath_ratio,
                path_length,
                n_heads,
                n_ffn_dense_layers,
                feat_drop=0.,
                attn_drop=0.,
                activation=nn.GELU()):
        super(sk_TripletTransformer, self).__init__(d_feats,
                d_hpath_ratio,
                path_length,
                n_heads,
                n_ffn_dense_layers,
                feat_drop,
                attn_drop,
                activation)
        self.node_out_adapter = SimpleAdapterLayer(d_feats,config.adapter_hidden_size)
        self.norm_adapter = SimpleAdapterLayer(d_feats,config.adapter_hidden_size)
        self.norm_adapter_norm = nn.LayerNorm(d_feats)
        self.node_out_adapter_norm = nn.LayerNorm(d_feats)

    def forward(self, g, triplet_h, dist_attn, path_attn):
        g = g.local_var()
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(
            len(g.edata['node_attn']), -1, 1)
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.ndata['hv'] = self.norm_adapter_norm(self.norm_adapter(g.ndata['hv']))
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1,
                                                                                                                    self.d_feats)
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        return self.node_out_adapter_norm(self.node_out_adapter(self.node_out_layer(triplet_h, g.ndata['agg_h'])))
    def make_finetune_trainable(self):
        self.node_out_adapter.requires_grad_(True)
        self.node_out_adapter_norm.requires_grad_(True)
        self.norm_adapter.requires_grad_(True)
        self.norm_adapter_norm.requires_grad_(True)

class sk_cross_Residual(sk_Residual):
    def __init__(self, config, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(sk_cross_Residual, self).__init__(config, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation)
        self.sk = sk_cross_adapter(d_out_feats, self.adapter_heads, self.adapter_heads_size, dropout=config.sk_dropout)

class sk_attncross_Residual(sk_Residual):
    def __init__(self, config, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(sk_attncross_Residual, self).__init__(config, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation)
        self.sk = sk_attncross_adapater(d_out_feats, self.adapter_heads, self.adapter_heads_size, dropout=config.sk_dropout)


class FineTunekgpt(nn.Module):
    def __init__(self, config):
        super(FineTunekgpt, self).__init__()
        self.config = config
        self.start_layer = config.start_layer
        self.model = LiGhT(
            d_node_feats=config.d_node_feats,
            d_edge_feats=config.d_edge_feats,
            d_g_feats=config.d_g_feats,
            d_fp_feats=config.d_fps,
            d_md_feats=config.d_mds,
            d_hpath_ratio=config.d_hpath_ratio,
            n_mol_layers=config.n_mol_layers,
            path_length=config.path_length,
            n_heads=config.n_heads,
            n_ffn_dense_layers=config.n_ffn_dense_layers,
            input_drop=0,
            attn_drop=config.GhT_dropout,
            feat_drop=config.GhT_dropout,
            n_node_types=config.vocab_size
        )

        del self.model.md_predictor
        del self.model.fp_predictor
        del self.model.node_predictor

    def replace_resuial_with_sk(self, config):
        for index, layer in enumerate(self.model.model.mol_T_layers):
            if index >= self.start_layer:
                named_modules = list(layer.named_modules())
                for name, module in named_modules:
                    if isinstance(module, Residual):
                        sk_res = sk_Residual(config=config, d_in_feats=module.in_proj.in_features,
                                             d_out_feats=module.in_proj.out_features,
                                             n_ffn_dense_layers=module.ffn.n_dense_layers, activation=module.ffn.act,
                                             feat_drop=module.feat_dropout.p)
                        sk_res.make_finetune_trainable()
                        setattr(layer, name, sk_res)

    def replace_resuial_with_sk_cross(self,config):
        for index, layer in enumerate(self.model.model.mol_T_layers):
            if index >= self.start_layer:
                named_modules = list(layer.named_modules())
                for name, module in named_modules:
                    if isinstance(module, Residual):
                        sk_res = sk_cross_Residual(config=config, d_in_feats=module.in_proj.in_features,
                                             d_out_feats=module.in_proj.out_features,
                                             n_ffn_dense_layers=module.ffn.n_dense_layers, activation=module.ffn.act,
                                             feat_drop=module.feat_dropout.p)
                        sk_res.make_finetune_trainable()
                        setattr(layer, name, sk_res)
    def replace_resuial_with_sk_attncross(self,config):
        for index, layer in enumerate(self.model.model.mol_T_layers):
            if index >= self.start_layer:
                named_modules = list(layer.named_modules())
                for name, module in named_modules:
                    if isinstance(module, Residual):
                        sk_res = sk_attncross_Residual(config=config, d_in_feats=module.in_proj.in_features,
                                             d_out_feats=module.in_proj.out_features,
                                             n_ffn_dense_layers=module.ffn.n_dense_layers, activation=module.ffn.act,
                                             feat_drop=module.feat_dropout.p)
                        sk_res.make_finetune_trainable()
                        setattr(layer, name, sk_res)

    def replace_with_norm_adapter(self,config):
        for index, layer in enumerate(self.model.model.mol_T_layers):
            if index >= self.start_layer:
                named_modules = list(layer.named_modules())
                for name, module in named_modules:
                    if isinstance(module, TripletTransformer):
                        norm_adapter = sk_TripletTransformer(config=config,d_feats=module.d_feats,d_hpath_ratio=module.d_hpath_ratio,path_length=module.path_length,
                                                             n_heads=module.n_heads,n_ffn_dense_layers=module.n_ffn_dense_layers,feat_drop=module.feat_dropout.p,
                                                             attn_drop=module.attn_dropout.p,activation=module.act)
                        norm_adapter.requires_grad_(False)
                        norm_adapter.make_finetune_trainable()
                        setattr(layer, name, norm_adapter)
    def add_adapter(self, key_word, config=None):
        # 冻结其他层的参数
        if config.add:
            for name, param in self.model.named_parameters():
                if 'predictor' not in name:
                    param.requires_grad = False
        if key_word == 'simple_adapter':
            adapter = SimpleAdapterLayer(self.model.model.n_heads,self.config.adapter_hidden_size)
            adapter.requires_grad_(True)
            self.model.model.dist_attn_layer.add_module('simple_adapter',adapter)
            adapter = SimpleAdapterLayer(self.model.model.n_heads,self.config.adapter_hidden_size)
            adapter.requires_grad_(True)
            self.model.model.path_attn_layer.add_module('simple_adapter',adapter)
            for index, layer in enumerate(self.model.model.mol_T_layers):
                if index >= self.start_layer:
                    adapter = SimpleAdapterLayer(layer.d_feats, self.config.adapter_hidden_size)
                    adapter.requires_grad_(True)
                    layer.add_module(key_word, adapter)
        elif key_word == 'sk':
            # adapter = sk_adapter(self.model.model.n_heads,1, self.config.sk_hidden_size,
            #                      dropout=0)
            # adapter.requires_grad_(True)
            # self.model.model.dist_attn_layer.add_module('sk',adapter)
            # adapter = sk_adapter(self.model.model.n_heads,1, self.config.sk_hidden_size,
            #                      dropout=0)
            # adapter.requires_grad_(True)
            # self.model.model.path_attn_layer.add_module('sk',adapter)
            self.replace_resuial_with_sk(config)
        elif key_word == 'sk_cross':
            # adapter = sk_adapter(self.model.model.n_heads, 1, self.config.sk_hidden_size,
            #                      dropout=0)
            # adapter.requires_grad_(True)
            # self.model.model.dist_attn_layer.add_module(key_word, adapter)
            # adapter = sk_adapter(self.model.model.n_heads, 1, self.config.sk_hidden_size,
            #                      dropout=0)
            # adapter.requires_grad_(True)
            # self.model.model.path_attn_layer.add_module(key_word, adapter)
            self.replace_resuial_with_sk_cross(config)
        elif key_word == 'sk_attncross':
            adapter = sk_adapter(self.model.model.n_heads, 1, self.config.sk_hidden_size,
                                 dropout=0)
            adapter.requires_grad_(True)
            self.model.model.dist_attn_layer.add_module(key_word, adapter)
            adapter = sk_adapter(self.model.model.n_heads, 1, self.config.sk_hidden_size,
                                 dropout=0)
            adapter.requires_grad_(True)
            self.model.model.path_attn_layer.add_module(key_word, adapter)
            self.replace_resuial_with_sk_attncross(config)
        elif key_word =='norm_adapter':
            self.replace_with_norm_adapter(config)

    def load_from_pretrain(self, model_path='Pretrain_checkpoints/KGPT.pth'):
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{model_path}').items()},
                                   strict=False)

    def forward(self, g, fp, md):
        return self.model.forward_tune(g, fp, md)
