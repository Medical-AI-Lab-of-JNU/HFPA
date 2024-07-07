from loralib import LoRALayer

import torch.nn.functional as F
from fast_transformers.masking import LengthMask as LM, FullMask, LengthMask
import loralib as lora
from functools import partial
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LayerNorm
import math
from MolFormer.rotate_attention.attention_layer import RotateAttentionLayer, RotaryEmbedding
from MolFormer.rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.events import EventDispatcher

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from adapters.adapters import *


class MolFormer(nn.Module):
    def __init__(self, tokenizer, config):
        super(MolFormer, self).__init__()
        self.tokenizer = tokenizer
        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        self.n_embd = config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, mask):
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        return x

class RotateAttention_simpleadapter0_Encoderlayer(TransformerEncoderLayer):
    def __init__(self,adapter_hidden,attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        self.d_ff = d_ff
        self.dropout = dropout
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.adapter_hidden = adapter_hidden
        if adapter_hidden != 0:
            self.adapter1 = SimpleAdapterLayer(d_model,adapter_hidden)
    def forward(self, x, attn_mask=None, length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        y = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        x = x + self.dropout(y)
        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        if self.adapter_hidden != 0:
            y = self.adapter1(y)
        return self.norm2(x+y)


class RotateAttention_simpleadapter_Encoderlayer(TransformerEncoderLayer):
    def __init__(self,adapter_hidden,attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        self.d_ff = d_ff
        self.dropout = dropout
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.adapter_hidden = adapter_hidden
        if adapter_hidden != 0:
            self.adapter1 = SimpleAdapterLayer(d_model,adapter_hidden)
            self.adapter2 = SimpleAdapterLayer(d_model, adapter_hidden)
    def forward(self, x, attn_mask=None, length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        y = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        if self.adapter_hidden != 0:
            y = self.adapter1(y)

        x = x + self.dropout(y)
        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        if self.adapter_hidden != 0:
            y = self.adapter2(y)
        return self.norm2(x+y)


class RotateAttention_hfpa_Encoderlayer(TransformerEncoderLayer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        self.d_ff = d_ff
        self.dropout = dropout
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return x+y, self.norm2



class RotateAttention_hfpa_Encoder(TransformerEncoder):
    def __init__(self, config, layers, norm_layer, event_dispatcher):
        super(RotateAttention_hfpa_Encoder, self).__init__(layers, norm_layer=norm_layer,
                                                                   event_dispatcher=event_dispatcher)
        self.config = config
        self.adapter_heads = self.config.heads  # heads的数目
        self.adapter_heads_size = int(self.config.hidden_size / self.adapter_heads)
        self.start_layer = config.start_layer
        self.sk_adapter = nn.ModuleList(
            [HFPA(self.config.n_embd, self.adapter_heads, self.adapter_heads_size,self.config.dropout) for _ in
             range(config.start_layer, self.config.n_layer + 1)])

    def forward(self, x, attn_mask=None, length_mask=None):
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for i,layer in enumerate(self.layers):
            x,layer_norm = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            if i >= self.start_layer:
                x = self.sk_adapter[i-self.start_layer](x,layer_norm)
            else:
                x = layer_norm(x)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

    def make_finetune_trainable(self):
        self.layers.requires_grad_(False)
        self.norm.requires_grad_(False)
        self.sk_adapter.requires_grad_(True)


class RotateAttention_hfpca_Encoder(RotateAttention_hfpa_Encoder):
    def __init__(self, config, layers, norm_layer, event_dispatcher):
        super(RotateAttention_hfpca_Encoder, self).__init__(config,layers, norm_layer=norm_layer,
                                                                   event_dispatcher=event_dispatcher)
        self.sk_adapter = nn.ModuleList(
            [HFPCA(self.config.n_embd, self.adapter_heads, self.adapter_heads_size,self.config.dropout) for _ in
             range(config.start_layer, self.config.n_layer + 1)])

class FineTuneMolFormer(nn.Module):
    def __init__(self, tokenizer, config):
        super(FineTuneMolFormer, self).__init__()
        self.config = config
        self.model = MolFormer(tokenizer, config)
        self.start_layer = config.start_layer

    def load_from_pretrain(self, pretrain_file='Pretrain_checkpoints/MolFormer.ckpt'):
        checkpoint = torch.load(pretrain_file)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)


    def replace_transformer_with_hfpa(self,config):
        self.model.requires_grad_(False)
        sk_encoder = RotateAttention_hfpa_Encoder(config, self.model.blocks.layers,
                                                        norm_layer=self.model.blocks.norm,
                                                        event_dispatcher=self.model.blocks.event_dispatcher)
        self.model.blocks = sk_encoder

        named_modules = list(self.model.blocks.named_modules())
        for name, module in named_modules:
            if isinstance(module, TransformerEncoderLayer):
                sk_transformer = RotateAttention_hfpa_Encoderlayer(attention=module.attention, d_model=module.linear1.in_features,
                                                                            d_ff=module.linear1.out_features, dropout=module.dropout.p,
                                                                            activation=module.activation, event_dispatcher=module.event_dispatcher)
                parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                parent_module = getattr(self.model.blocks, parent_name)  # 获取父模块对象
                setattr(parent_module, name.split('.')[-1], sk_transformer)  # 替换为LoRA变体层
        self.model.blocks.make_finetune_trainable()


    def replace_transformer_with_hfpca(self,config):
        self.model.requires_grad_(False)
        sk_encoder = RotateAttention_hfpca_Encoder(config, self.model.blocks.layers,
                                                        norm_layer=self.model.blocks.norm,
                                                        event_dispatcher=self.model.blocks.event_dispatcher)
        self.model.blocks = sk_encoder
        named_modules = list(self.model.blocks.named_modules())
        for name, module in named_modules:
            if isinstance(module, TransformerEncoderLayer):
                sk_transformer = RotateAttention_hfpa_Encoderlayer(attention=module.attention, d_model=module.linear1.in_features,
                                                                            d_ff=module.linear1.out_features, dropout=module.dropout.p,
                                                                            activation=module.activation, event_dispatcher=module.event_dispatcher)
                parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                parent_module = getattr(self.model.blocks, parent_name)  # 获取父模块对象
                setattr(parent_module, name.split('.')[-1], sk_transformer)  # 替换为LoRA变体层
        self.model.blocks.make_finetune_trainable()

    def replace_transformer_with_simpleadapter(self,config):
        self.model.requires_grad_(False)
        for index, layer in enumerate(self.model.blocks.layers):
            if index >= self.start_layer:
                simple_adapter = RotateAttention_simpleadapter_Encoderlayer(config.adapter_hidden_size,layer.attention,layer.linear1.in_features,
                                                                            d_ff=layer.linear1.out_features,dropout=layer.dropout.p,
                                                                            activation=layer.activation,event_dispatcher=layer.event_dispatcher)
                self.model.blocks.layers[index] = simple_adapter
                self.model.blocks.layers[index].requires_grad_(False)
                self.model.blocks.layers[index].adapter1.requires_grad_(True)
                self.model.blocks.layers[index].adapter2.requires_grad_(True)

    def replace_transformer_with_simpleadapter0(self,config):
        self.model.requires_grad_(False)
        for index, layer in enumerate(self.model.blocks.layers):
            if index >= self.start_layer:
                simple_adapter = RotateAttention_simpleadapter0_Encoderlayer(config.adapter_hidden_size,layer.attention,layer.linear1.in_features,
                                                                            d_ff=layer.linear1.out_features,dropout=layer.dropout.p,
                                                                            activation=layer.activation,event_dispatcher=layer.event_dispatcher)
                self.model.blocks.layers[index] = simple_adapter
                self.model.blocks.layers[index].requires_grad_(False)
                self.model.blocks.layers[index].adapter1.requires_grad_(True)


    def add_adapter(self, key_word, config=None):
        if config.add:
            state_dict = self.model.state_dict()
            for param_name, param in state_dict.items():
                param.requires_grad = False

        if key_word == 'simple_adapter':
            self.model.requires_grad_(False)
            self.replace_transformer_with_simpleadapter0(config=config)
        elif key_word == 'norm_adapter':
            self.replace_transformer_with_simpleadapter(config=config)
        elif key_word == 'hfpa':
            self.replace_transformer_with_hfpa(self.config)
        elif key_word == 'hfpca':
            self.replace_transformer_with_hfpca(self.config)

    def forward(self, idx, mask):
        x = self.model(idx, mask)
        return x
