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
from dataclasses import dataclass, field
from fast_transformers.events import EventDispatcher

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from adapters.adapters import *
from visualization.MolFormer.rotate_attention.transformers import VizEncoderLayer


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
            query_dimensions=config.n_embd // config.n_head,
            value_dimensions=config.n_embd // config.n_head,
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


class RotateAttention_lora_Layer(RotateAttentionLayer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher="", r=0, alpha=1, dropout=0, enable=None):
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        if enable is None:
            enable = [True, False, True]
        self.inner_attention = attention
        self.lora_config = LoRAConfig(r, alpha, dropout, enable)
        assert d_model % n_heads == 0
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        if self.lora_config.enable[0]:
            self.query_projection = lora.Linear(d_model, d_keys * n_heads, r=self.lora_config.r,
                                                lora_alpha=self.lora_config.alpha,
                                                lora_dropout=self.lora_config.dropout)
        else:
            self.query_projection = Linear(d_model, d_keys * n_heads)
        if self.lora_config.enable[1]:
            self.key_projection = lora.Linear(d_model, d_keys * n_heads, r=self.lora_config.r,
                                              lora_alpha=self.lora_config.alpha, lora_dropout=self.lora_config.dropout)
        else:
            self.key_projection = Linear(d_model, d_keys * n_heads)
        if self.lora_config.enable[2]:
            self.value_projection = lora.Linear(d_model, d_values * n_heads, r=self.lora_config.r,
                                                lora_alpha=self.lora_config.alpha,
                                                lora_dropout=self.lora_config.dropout)
        else:
            self.value_projection = Linear(d_model, d_values * n_heads)
        self.rotaryemb = RotaryEmbedding(d_keys)
        if self.lora_config.enable[3]:
            self.out_projection = lora.Linear(d_values * n_heads, d_model, r=self.lora_config.r,
                                              lora_alpha=self.lora_config.alpha, lora_dropout=self.lora_config.dropout)
        else:
            self.out_projection = Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)



class RotateAttention_skadapter_Encoderlayer(TransformerEncoderLayer):
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



class RotateAttention_hfpadapter_Encoderlayer(TransformerEncoderLayer):
    def __init__(self, heads,hidden_size,sk_dropout,attention, d_model, d_ff=None, dropout=0.1,
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
        self.adapter_heads_size = int(hidden_size / heads)

        if hidden_size != 0:
            self.adapter1 = sk_adapter(d_model, heads, self.adapter_heads_size,sk_dropout)
            #self.adapter_gate = nn.Sequential(nn.Linear(d_model,1),nn.Sigmoid())
            self.adapter2 = sk_adapter(d_model, heads, self.adapter_heads_size,sk_dropout)

    def forward(self, x, attn_mask=None, length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        y =self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        if self.adapter_heads_size != 0:
            #y_temp = self.adapter1(y, resiual=False)
            #y = y + self.adapter_gate(y_temp)*y_temp
            y = y+self.adapter1(y, resiual=False)

        x = x + self.dropout(y)
        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        if self.adapter_heads_size != 0:
            y = self.adapter2(y)
        return x + y

class RotateAttention_hfpcross_Encoderlayer(RotateAttention_hfpadapter_Encoderlayer):
    def __init__(self, heads,hidden_size,sk_dropout,attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(RotateAttention_hfpcross_Encoderlayer, self).__init__(heads,hidden_size,sk_dropout,attention, d_model,
                                                                    d_ff, dropout,activation, event_dispatcher)
        if hidden_size != 0:
            self.adapter1 = sk_cross_adapter(d_model, heads, self.adapter_heads_size, sk_dropout)
            self.adapter_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
            self.adapter2 = sk_cross_adapter(d_model, heads, self.adapter_heads_size, sk_dropout)

class RotateAttention_hfpattncross_Encoderlayer(RotateAttention_hfpadapter_Encoderlayer):
    def __init__(self, heads,hidden_size,sk_dropout,attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(RotateAttention_hfpattncross_Encoderlayer, self).__init__(heads,hidden_size,sk_dropout,attention, d_model,
                                                                    d_ff, dropout,activation, event_dispatcher)
        if hidden_size != 0:
            self.adapter1 = sk_attncross_adapater(d_model, heads, self.adapter_heads_size, sk_dropout)
            self.adapter2 = sk_attncross_adapater(d_model, heads, self.adapter_heads_size, sk_dropout)



class RotateAttention_skip_apapter_Encoder(TransformerEncoder):
    def __init__(self, config, layers, norm_layer, event_dispatcher):
        super(RotateAttention_skip_apapter_Encoder, self).__init__(layers, norm_layer=norm_layer,
                                                                   event_dispatcher=event_dispatcher)
        self.config = config
        self.adapter_heads = self.config.sk_heads  # heads的数目
        self.adapter_heads_size = int(self.config.sk_hidden_size / self.adapter_heads)
        self.start_layer = config.start_layer
        self.sk_adapter = nn.ModuleList(
            [sk_adapter(self.config.n_embd, self.adapter_heads, self.adapter_heads_size,self.config.sk_dropout) for _ in
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


class RotateAttention_sk_cross_apapter_Encoder(RotateAttention_skip_apapter_Encoder):
    def __init__(self, config, layers, norm_layer, event_dispatcher):
        super(RotateAttention_sk_cross_apapter_Encoder, self).__init__(config,layers, norm_layer=norm_layer,
                                                                   event_dispatcher=event_dispatcher)
        self.sk_adapter = nn.ModuleList(
            [sk_cross_adapter(self.config.n_embd, self.adapter_heads, self.adapter_heads_size,self.config.sk_dropout) for _ in
             range(config.start_layer, self.config.n_layer + 1)])

class RotateAttention_sk_attncross_apapter_Encoder(RotateAttention_skip_apapter_Encoder):
    def __init__(self, config, layers, norm_layer, event_dispatcher):
        super(RotateAttention_sk_attncross_apapter_Encoder, self).__init__(config,layers, norm_layer=norm_layer,
                                                                   event_dispatcher=event_dispatcher)
        self.sk_adapter = nn.ModuleList(
            [sk_attncross_adapater(self.config.n_embd, self.adapter_heads, self.adapter_heads_size,self.config.sk_dropout) for _ in
             range(config.start_layer, self.config.n_layer + 1)])

class RotateAttention_prompt_Encoder(TransformerEncoder):
    def __init__(self, prompt_config, layers, norm_layer, event_dispatcher):
        super(RotateAttention_prompt_Encoder, self).__init__(layers, norm_layer=norm_layer,
                                                             event_dispatcher=event_dispatcher)
        self.prompt_config = prompt_config

        num_tokens = self.prompt_config.num_tokens
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.prompt_dropout)

        # if project the prompt embeddings
        if self.prompt_config.project > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.project
            self.prompt_proj = nn.Linear(
                prompt_dim, prompt_config.prompt_hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = prompt_config.hidden_size
            self.prompt_proj = nn.Identity()

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.kaiming_normal_(self.prompt_embeddings.data)

        if self.prompt_config.DEEP:  # noqa

            total_d_layer = len(self.layers) - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.kaiming_normal_(self.deep_prompt_embeddings.data)

    def incorporate_prompt(self, x):

        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def mark_only_prompt_as_trainable(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.norm.requires_grad_(False)
            self.layers.requires_grad_(False)
            self.prompt_proj.requires_grad_(True)
            self.prompt_dropout.requires_grad_(True)
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, x, attn_mask=None, length_mask=None):

        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(N, -1, -1))
                    x = torch.cat((
                        x[:, :1, :],
                        deep_prompt_emb,
                        x[:, (1 + self.num_tokens):, :]
                    ), dim=1)
                    x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
                # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

    def forward(self, x, attn_mask=None, length_mask=None):
        # this is the default version:
        x = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            x = self.forward_deep_prompt(x, attn_mask=None, length_mask=None)
        else:
            x = super().forward(x, attn_mask=None, length_mask=None)
        return x



class RotateAttention_mutilora_Layer(RotateAttentionLayer):

    def __init__(self, attention, d_model, n_heads, config, d_keys=None,
                 d_values=None, event_dispatcher=""):
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        self.enable = config.muti_enable
        if config.muti_enable is None:
            self.enable = [True, False, True, False]
        self.inner_attention = attention

        self.r = config.muti_r
        self.alpha = config.muti_alpha
        self.dropout = config.muti_dropout
        self.muti_heads = config.muti_lora_heads
        self.muti_factor = config.muti_lora_factor
        assert d_model % n_heads == 0
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        if self.enable[0]:
            self.query_projection = MutiLoraLinear(d_model, d_keys * n_heads, r=self.r,
                                                   lora_alpha=self.alpha,
                                                   lora_dropout=self.dropout,
                                                   lora_heads=self.muti_heads,
                                                   lora_factor=self.muti_factor)
        else:
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        if self.enable[1]:
            self.key_projection = MutiLoraLinear(d_model, d_keys * n_heads, r=self.r,
                                                 lora_alpha=self.alpha,
                                                 lora_dropout=self.dropout,
                                                 lora_heads=self.muti_heads,
                                                 lora_factor=self.muti_factor)
        else:
            self.key_projection = Linear(d_model, d_keys * n_heads)
        if self.enable[2]:
            self.value_projection = MutiLoraLinear(d_model, d_values * n_heads, r=self.r,
                                                   lora_alpha=self.alpha,
                                                   lora_dropout=self.dropout,
                                                   lora_heads=self.muti_heads,
                                                   lora_factor=self.muti_factor)
        else:
            self.value_projection = Linear(d_model, d_values * n_heads)
        self.rotaryemb = RotaryEmbedding(d_keys)
        if self.enable[3]:
            self.out_projection = MutiLoraLinear(d_values * n_heads, d_model, r=self.r,
                                                 lora_alpha=self.alpha,
                                                 lora_dropout=self.dropout,
                                                 lora_heads=self.muti_heads,
                                                 lora_factor=self.muti_factor)
        else:
            self.out_projection = Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)



class FineTuneMolFormer(nn.Module):
    def __init__(self, tokenizer, config):
        super(FineTuneMolFormer, self).__init__()
        self.config = config
        self.model = MolFormer(tokenizer, config)
        self.start_layer = config.start_layer

    def load_from_pretrain(self, pretrain_file='Pretrain_checkpoints/MolFormer.ckpt'):
        checkpoint = torch.load(pretrain_file)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def replace_transformer_with_lora(self, r, alpha, dropout, enable):
        named_modules = list(self.model.blocks.layers.named_modules())

        index = 0
        for name, module in named_modules:
            if isinstance(module, RotateAttentionLayer):
                if index >= self.start_layer:
                    lora_transformer = RotateAttention_lora_Layer(attention=module.attention, d_model=module.d_model,
                                                                  n_heads=module.n_heads, d_keys=module.d_keys,
                                                                  d_values=module.d_values,
                                                                  event_dispatcher=module.event_dispatcher,
                                                                  r=r, alpha=alpha, dropout=dropout,
                                                                  enable=enable)
                    parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                    parent_module = getattr(self.model.blocks.layers, parent_name)  # 获取父模块对象
                    setattr(parent_module, name.split('.')[-1], lora_transformer)  # 替换为LoRA变体层
                index = index + 1

    def replace_transformer_with_mutilora(self, config):
        named_modules = list(self.model.blocks.layers.named_modules())

        index = 0
        for name, module in named_modules:
            if isinstance(module, RotateAttentionLayer):
                if index >= 2:
                    lora_transformer = RotateAttention_mutilora_Layer(attention=module.attention,
                                                                      d_model=module.d_model,
                                                                      n_heads=module.n_heads, d_keys=module.d_keys,
                                                                      d_values=module.d_values,
                                                                      event_dispatcher=module.event_dispatcher,
                                                                      config=config)
                    parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                    parent_module = getattr(self.model.blocks.layers, parent_name)  # 获取父模块对象
                    setattr(parent_module, name.split('.')[-1], lora_transformer)  # 替换为LoRA变体层
                index = index + 1

    def replace_transformer_with_prompt(self, prompt_config):
        prompt_encoder = RotateAttention_prompt_Encoder(prompt_config, self.model.blocks.layers,
                                                        norm_layer=self.model.blocks.norm,
                                                        event_dispatcher=self.model.blocks.event_dispatcher)
        prompt_encoder.mark_only_prompt_as_trainable(True)

        self.model.blocks = prompt_encoder

    def replace_transformer_with_sk(self,config):
        sk_encoder = RotateAttention_skip_apapter_Encoder(config, self.model.blocks.layers,
                                                        norm_layer=self.model.blocks.norm,
                                                        event_dispatcher=self.model.blocks.event_dispatcher)
        self.model.blocks = sk_encoder

        named_modules = list(self.model.blocks.named_modules())
        for name, module in named_modules:
            if isinstance(module, TransformerEncoderLayer):
                sk_transformer = RotateAttention_skadapter_Encoderlayer(attention=module.attention, d_model=module.linear1.in_features,
                                                                            d_ff=module.linear1.out_features, dropout=module.dropout.p,
                                                                            activation=module.activation, event_dispatcher=module.event_dispatcher)
                parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                parent_module = getattr(self.model.blocks, parent_name)  # 获取父模块对象
                setattr(parent_module, name.split('.')[-1], sk_transformer)  # 替换为LoRA变体层
        self.model.blocks.make_finetune_trainable()


    def replace_transformer_with_sk_corss(self,config):
        sk_encoder = RotateAttention_sk_cross_apapter_Encoder(config, self.model.blocks.layers,
                                                        norm_layer=self.model.blocks.norm,
                                                        event_dispatcher=self.model.blocks.event_dispatcher)
        self.model.blocks = sk_encoder
        named_modules = list(self.model.blocks.named_modules())
        for name, module in named_modules:
            if isinstance(module, TransformerEncoderLayer):
                sk_transformer = RotateAttention_skadapter_Encoderlayer(attention=module.attention, d_model=module.linear1.in_features,
                                                                            d_ff=module.linear1.out_features, dropout=module.dropout.p,
                                                                            activation=module.activation, event_dispatcher=module.event_dispatcher)
                parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                parent_module = getattr(self.model.blocks, parent_name)  # 获取父模块对象
                setattr(parent_module, name.split('.')[-1], sk_transformer)  # 替换为LoRA变体层
        self.model.blocks.make_finetune_trainable()

    def replace_transformer_with_sk_attncorss(self,config):
        sk_encoder = RotateAttention_sk_attncross_apapter_Encoder(config, self.model.blocks.layers,
                                                        norm_layer=self.model.blocks.norm,
                                                        event_dispatcher=self.model.blocks.event_dispatcher)
        self.model.blocks = sk_encoder
        named_modules = list(self.model.blocks.named_modules())
        for name, module in named_modules:
            if isinstance(module, TransformerEncoderLayer):
                sk_transformer = RotateAttention_skadapter_Encoderlayer(attention=module.attention, d_model=module.linear1.in_features,
                                                                            d_ff=module.linear1.out_features, dropout=module.dropout.p,
                                                                            activation=module.activation, event_dispatcher=module.event_dispatcher)
                parent_name, _ = name.rsplit('.', 1)  # 获取父模块的名称
                parent_module = getattr(self.model.blocks, parent_name)  # 获取父模块对象
                setattr(parent_module, name.split('.')[-1], sk_transformer)
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

    def replace_transformer_with_hfpadapter(self, config):
        self.model.requires_grad_(False)
        for index, layer in enumerate(self.model.blocks.layers):
            if index >= self.start_layer:
                simple_adapter = RotateAttention_hfpadapter_Encoderlayer(config.sk_heads, config.sk_hidden_size,config.sk_dropout,
                                                                        layer.attention,
                                                                        layer.linear1.in_features,
                                                                        d_ff=layer.linear1.out_features,
                                                                        dropout=layer.dropout.p,
                                                                        activation=layer.activation,
                                                                        event_dispatcher=layer.event_dispatcher)
                self.model.blocks.layers[index] = simple_adapter
                self.model.blocks.layers[index].requires_grad_(False)
                self.model.blocks.layers[index].adapter1.requires_grad_(True)
                #self.model.blocks.layers[index].adapter_gate.requires_grad_(True)
                self.model.blocks.layers[index].adapter2.requires_grad_(True)

    def replace_transformer_with_hfpcadapter(self, config):
        self.model.requires_grad_(False)
        for index, layer in enumerate(self.model.blocks.layers):
            if index >= self.start_layer:
                simple_adapter = RotateAttention_hfpcross_Encoderlayer(config.sk_heads, config.sk_hidden_size,config.sk_dropout,
                                                                        layer.attention,
                                                                        layer.linear1.in_features,
                                                                        d_ff=layer.linear1.out_features,
                                                                        dropout=layer.dropout.p,
                                                                        activation=layer.activation,
                                                                        event_dispatcher=layer.event_dispatcher)
                self.model.blocks.layers[index] = simple_adapter
                self.model.blocks.layers[index].requires_grad_(False)
                self.model.blocks.layers[index].adapter1.requires_grad_(True)
                self.model.blocks.layers[index].adapter2.requires_grad_(True)
    def replace_transformer_with_hfpcaadapter(self, config):
        self.model.requires_grad_(False)
        for index, layer in enumerate(self.model.blocks.layers):
            if index >= self.start_layer:
                simple_adapter = RotateAttention_hfpattncross_Encoderlayer(config.sk_heads, config.sk_hidden_size,config.sk_dropout,
                                                                        layer.attention,
                                                                        layer.linear1.in_features,
                                                                        d_ff=layer.linear1.out_features,
                                                                        dropout=layer.dropout.p,
                                                                        activation=layer.activation,
                                                                        event_dispatcher=layer.event_dispatcher)
                self.model.blocks.layers[index] = simple_adapter
                self.model.blocks.layers[index].requires_grad_(False)
                self.model.blocks.layers[index].adapter1.requires_grad_(True)
                self.model.blocks.layers[index].adapter2.requires_grad_(True)
    def add_adapter(self, key_word, config=None):
        # 冻结其他层的参数
        if config.add:
            for param in self.model.parameters():
                param.requires_grad = False

        if key_word == 'simple_adapter':
            for index, layer in enumerate(self.model.blocks.layers):
                if index >= self.start_layer:
                    adapter = SimpleAdapterLayer(self.model.n_embd, self.config.adapter_hidden_size)
                    adapter.requires_grad_(True)
                    layer.add_module(key_word, adapter)
        elif key_word == 'lora':
            self.replace_transformer_with_lora(r=config.lora_r, alpha=config.lora_alpha, dropout=config.lora_dropout,
                                               enable=config.enable)
            lora.mark_only_lora_as_trainable(self.model, 'lora_only')
        elif key_word == 'prompt':
            self.replace_transformer_with_prompt(prompt_config=config)
        elif key_word == 'norm_adapter':
            self.replace_transformer_with_simpleadapter(config=config)
        elif key_word == 'hfpadapter':
            self.replace_transformer_with_hfpadapter(config = self.config)
        elif key_word == 'hfpcadapter':
            self.replace_transformer_with_hfpcadapter(config = self.config)
        elif key_word == 'hfpcaadapter':
            self.replace_transformer_with_hfpcaadapter(config = self.config)
        elif key_word == 'sk':
            self.replace_transformer_with_sk(self.config)
        elif key_word == 'sk_cross':
            self.replace_transformer_with_sk_corss(self.config)
        elif key_word == 'sk_attncross':
            self.replace_transformer_with_sk_attncorss(self.config)

        # elif key_word == 'mutiadapter':
        #     for index, layer in enumerate(self.model.blocks.layers):
        #         if index >= self.start_layer and index < len(self.model.blocks.layers) - 1:
        #             adapter = MutiAdapter(self.model.n_embd, self.config.muti_heads_hidden_size,
        #                                   self.config.head_factor, self.config.muti_heads, self.config.head_dropout)
        #             adapter.requires_grad_(True)
        #             layer.add_module(key_word, adapter)

        # elif key_word == 'lora_mutiadapter':
        #     self.replace_transformer_with_lora(r=config.lora_r, alpha=config.lora_alpha, dropout=config.lora_dropout,
        #                                        enable=config.enable)
        #     lora.mark_only_lora_as_trainable(self.model, 'lora_only')
        #     for index, layer in enumerate(self.model.blocks.layers):
        #         if index >= self.start_layer:
        #             adapter = MutiAdapter(self.model.n_embd, self.config.muti_heads_hidden_size,
        #                                   self.config.head_factor,
        #                                   self.config.muti_heads, self.config.head_dropout)
        #             adapter.requires_grad_(True)
        #             layer.add_module(key_word, adapter)
    def forward(self, idx, mask):
        x = self.model(idx, mask)
        return x
