from argparse import Namespace

import torch
from fast_transformers.masking import LengthMask as LM, FullMask, LengthMask
from functools import partial
from torch.nn import Linear, Dropout, LayerNorm

from visualization.MolFormer.get_attention_map_full import get_bert
from visualization.MolFormer.rotate_attention.attention_layer import RotateAttentionLayer, RotaryEmbedding
from visualization.MolFormer.rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.events import EventDispatcher
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from adapters.adapters import *
from visualization.MolFormer.rotate_attention.transformers import VizEncoder,VizEncoderLayer


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



class RotateAttention_skadapter_Encoderlayer(VizEncoderLayer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(VizEncoderLayer, self).__init__(attention, d_model, d_ff=None, dropout=0.1,
                 activation=activation, event_dispatcher=event_dispatcher)
        self.d_ff = d_ff
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
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        out, attention_mask = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        x = x + self.dropout(out)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return x+y, self.norm2, attention_mask.detach()


class RotateAttention_skip_apapter_Encoder(VizEncoder):
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
        attention_mask_list = []
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
                      LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for i,layer in enumerate(self.layers):
            x,layer_norm,attention_mask = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            if i >= self.start_layer:
                x = self.sk_adapter[i-self.start_layer](x,layer_norm)
            else:
                x = layer_norm(x)
            attention_mask_list.append(attention_mask)
        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x,attention_mask_list

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



class RotateAttention_simpleadapter_Encoderlayer(VizEncoderLayer):
    def __init__(self,adapter_hidden,attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(VizEncoderLayer, self).__init__(attention, d_model, d_ff=None, dropout=0.1,
                                              activation=activation, event_dispatcher=event_dispatcher)
        self.d_ff = d_ff
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
        y, attention_mask = self.attention(
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
        return self.norm2(x+y),attention_mask.detach()


class FineTuneMolFormer(nn.Module):
    def __init__(self, tokenizer, config):
        super(FineTuneMolFormer, self).__init__()
        self.start_layer = None
        self.config = config
        self.model = get_bert(config,tokenizer)
        

    def load_fine_tune(self,fine_tune_root="checkpoints/MolFormer/bbbp/sk/version_9/epoch=3-avg_train_metric=0.926-avg_val_metric=0.9179.ckpt"):
        if fine_tune_root != None:
            fine_model = torch.load(fine_tune_root)
            config = Namespace(**fine_model["hyper_parameters"])
            self.start_layer = config.start_layer
            self.add_adapter(key_word=config.model_type,config=config)
            tmp_model = torch.load(self.config.pretrain_model_root)["state_dict"]

            pretrained_params = fine_model['state_dict']

            for pretrained_name, value in list(pretrained_params.items()):
                model_name = pretrained_name.replace('model.model.', '')
                pretrained_params[model_name] = pretrained_params.pop(pretrained_name)
                if pretrained_name.startswith('net'):
                    pretrained_params.pop(pretrained_name)
            pretrained_params.update(tmp_model)
            loaded_state_dict = self.model.load_state_dict(pretrained_params,strict=False)
            print("missing keys and unexpected keys:")
            for name in loaded_state_dict:
                print(name)
        else:
            tmp_model = torch.load('Pretrain_checkpoints/MolFormer.ckpt')["state_dict"]
            self.model.load_state_dict(tmp_model,strict=True)
        self.model = self.model.eval()

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
                setattr(parent_module, name.split('.')[-1], sk_transformer)  # 替换为LoRA变体层

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


    def add_adapter(self, key_word, config=None):
        if key_word == 'simple_adapter':
            for index, layer in enumerate(self.model.blocks.layers):
                if index >= self.start_layer:
                    adapter = SimpleAdapterLayer(self.model.n_embd, self.config.adapter_hidden_size)
                    adapter.requires_grad_(False)
                    layer.add_module(key_word, adapter)
        elif key_word == 'lora':
            self.replace_transformer_with_lora(r=config.lora_r, alpha=config.lora_alpha, dropout=config.lora_dropout,
                                               enable=config.enable)
            lora.mark_only_lora_as_trainable(self.model, 'lora_only')
        elif key_word == 'prompt':
            self.replace_transformer_with_prompt(prompt_config=config)
        elif key_word == 'norm_adapter':
            self.replace_transformer_with_simpleadapter(config=config)
        elif key_word == 'mutiadapter':
            for index, layer in enumerate(self.model.blocks.layers):
                if index >= self.start_layer and index < len(self.model.blocks.layers) - 1:
                    adapter = MutiAdapter(self.model.n_embd, self.config.muti_heads_hidden_size,
                                          self.config.head_factor, self.config.muti_heads, self.config.head_dropout)
                    adapter.requires_grad_(True)
                    layer.add_module(key_word, adapter)
        elif key_word == 'lora_muti':
            self.replace_transformer_with_mutilora(config)
            lora.mark_only_lora_as_trainable(self.model, 'lora_only')
        elif key_word == 'lora_mutiadapter':
            self.replace_transformer_with_lora(r=config.lora_r, alpha=config.lora_alpha, dropout=config.lora_dropout,
                                               enable=config.enable)
            lora.mark_only_lora_as_trainable(self.model, 'lora_only')
            for index, layer in enumerate(self.model.blocks.layers):
                if index >= self.start_layer:
                    adapter = MutiAdapter(self.model.n_embd, self.config.muti_heads_hidden_size,
                                          self.config.head_factor,
                                          self.config.muti_heads, self.config.head_dropout)
                    adapter.requires_grad_(True)
                    layer.add_module(key_word, adapter)
        elif key_word == 'sk':
            self.replace_transformer_with_sk(config)
        elif key_word == 'sk_cross':
            self.replace_transformer_with_sk_corss(config)
        elif key_word == 'sk_attncross':
            self.replace_transformer_with_sk_attncorss(config)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, idx, mask,mode):
        x = self.model(idx, mask,mode=mode)
        return x
