# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb


class NewsEncoder(nn.Module):

    def __init__(self, opt, dropout_rate=0.2):
        super(NewsEncoder, self).__init__()
        self.opt = opt

        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.w2v_path)).float(), freeze=False)

        input_size = opt.user_dim
        self.dim_tran = nn.Linear(opt.word_dim, opt.user_dim)
        nn.init.uniform_(self.dim_tran.weight, -0.1, 0.1)
        self.dropout = nn.Dropout(opt.drop_out)
        self.category_encoder = CategoryEncoder(category_embed=opt.category_embed, category_pad_token_id=0, dropout=opt.drop_out, output_dim=opt.cat_dim)

        if opt.word_att:
            self.w_attn = VA_Atten(opt.user_dim, opt.attn_dim)

        self.tr = nn.MultiheadAttention(embed_dim=input_size, num_heads=20)


    def forward(self, title, cat, sub, training=1):
        cat_emb = self.category_encoder(cat)
        if training == 1:
            B, C, D = title.shape
            title = title.view(B*C, D)
        title_emb = self.word_emb(title)  # [B, K, Seq_t, H]
        inputs = self.dim_tran(self.dropout(title_emb))
        title_en = (self.tr(inputs.permute(1, 0, 2), inputs.permute(1, 0, 2), inputs.permute(1, 0, 2))[0]).permute(1, 0, 2)
        if self.opt.word_att:
            title_repr = self.w_attn(title_en)
        else:
            title_repr = title_en.mean(-2)
        if training == 1:
            return title_repr.view(B, C, -1), cat_emb
        else:
            return title_repr, cat_emb



class UserEncoder(nn.Module):
    def __init__(self, opt, user_dim, dropout_rate=0.2):
        super(UserEncoder, self).__init__()
        self.opt = opt
        if self.opt.news_att:
            self.n_attn = VA_Atten(user_dim, opt.attn_dim)

        self.tr = nn.MultiheadAttention(embed_dim=user_dim, num_heads=20)

        self.dropout = nn.Dropout(opt.drop_out)

    def forward(self, news_fea):
        news_fea = self.dropout(news_fea) # [B, K, d]

        user_fea = self.tr(news_fea.permute(1, 0, 2), news_fea.permute(1, 0, 2),
                           news_fea.permute(1, 0, 2))[0]

        user_fea = self.dropout(user_fea.permute(1, 0, 2))
        if self.opt.news_att:
            user_fea = self.n_attn(user_fea)  # [B, d]
        else:
            user_fea = user_fea.mean(1)
        return user_fea


class VA_Atten(nn.Module):

    def __init__(self, dim1, dim2):
        super().__init__()
        self.drop_out = nn.Dropout(0.2)
        self.att_fc = nn.Linear(dim1, dim2)
        self.att_h = nn.Linear(dim2, 1)
        nn.init.xavier_uniform_(self.att_fc.weight, gain=1)
        nn.init.uniform_(self.att_h.weight, -0.1, 0.1)

    def forward(self, x):

        score = self.att_h(torch.tanh(self.att_fc(x)))
        weight = F.softmax(score, -2)
        return (x*weight).sum(-2)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pe.weight, -0.1, 0.1)

    def forward(self, x):
        b, l, d = x.size()
        seq_len = torch.arange(l).to(x.device)
        return x + self.pe(seq_len).unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CategoryEncoder(nn.Module):
    def __init__(self, category_embed, category_pad_token_id, dropout, output_dim):
        super().__init__()
        category_embedding = torch.from_numpy(np.load(category_embed)).float()
        self.category_encoder = nn.Embedding.from_pretrained(category_embedding, freeze=False,
                                                                       padding_idx=category_pad_token_id)
        self.category_embed_dim = category_embedding.shape[1]
        self.reduce_dim = nn.Linear(in_features=self.category_embed_dim, out_features=output_dim)
        self.cat_embed_dropout = nn.Dropout(dropout)

    def forward(self, categories):
        category_emb = self.category_encoder(categories)
        category_repr = self.reduce_dim(category_emb)
        category_repr = self.cat_embed_dropout(category_repr)