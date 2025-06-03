from abc import ABC
from typing import Union
import ipdb
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, BertModel, BertPreTrainedModel, BertConfig, AutoModel
from transformers import AutoTokenizer



class BertEncoder(nn.Module):
    def __init__(self, opt):

        super(BertEncoder, self).__init__()
        # ipdb.set_trace()
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.pretrain_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.pretrain_model_path)

        self.reduce_dim = nn.Linear(in_features=opt.bert_dim, out_features=opt.user_dim)
        self.word_embed_dropout = nn.Dropout(opt.drop_out)
        self._embed_dim = opt.user_dim

        self.category_encoder = CategoryEncoder(category_embed=opt.category_embed, category_pad_token_id=0, dropout=opt.drop_out, output_dim=opt.cat_dim)


    def forward(self, title, cat, sub, training=1):
        cat_emb = self.category_encoder(cat)
        if training == 1:
            B, C, D = title.shape
            title = title.view(B*C, D)
        title_mask = (title != self.tokenizer.pad_token_id)
        title_repr = self.bert(input_ids=title, attention_mask=title_mask)[1]
        title_repr = self.reduce_dim(title_repr)
        title_repr = self.word_embed_dropout(title_repr)
        if training == 1:
            return title_repr.view(B, C, -1), cat_emb
        else:
            return title_repr, cat_emb

class CategoryEncoder(nn.Module):
    def __init__(self, category_embed, category_pad_token_id, dropout, output_dim):
        super().__init__()
        self.category_encoder = nn.Embedding(25, 300, padding_idx=category_pad_token_id) 
        self.category_embed_dim = 300 
        self.reduce_dim = nn.Linear(in_features=self.category_embed_dim, out_features=output_dim)
        self.cat_embed_dropout = nn.Dropout(dropout)

    def forward(self, categories):
        category_emb = self.category_encoder(categories)
        category_repr = self.reduce_dim(category_emb)
        category_repr = self.cat_embed_dropout(category_repr)