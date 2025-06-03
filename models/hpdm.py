# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from torch.autograd import Variable
from .encoder import NewsEncoder, UserEncoder
from .news_encoder import BertEncoder


class HPDM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.out_dim = opt.user_dim + opt.cat_dim

        if self.opt.enc == 'glove':
            self.news_encoder = NewsEncoder(opt)
        else:
            self.news_encoder = BertEncoder(opt)
        self.mode = opt.mode

        self.user_encoder_pop = UserEncoder(opt, opt.user_dim)
        self.ucat_encoder_pop = UserEncoder(opt, opt.cat_dim)

        self.linear_news_pop = nn.Linear(opt.user_dim, 1)
        self.linear_cat_pop = nn.Linear(opt.cat_dim, 1)
        self.user_encoder_unpop = UserEncoder(opt, opt.user_dim)
        self.ucat_encoder_unpop = UserEncoder(opt, opt.cat_dim)
        self.linear_news_unpop = nn.Linear(opt.user_dim, 1)
        self.linear_cat_unpop = nn.Linear(opt.cat_dim, 1)


        self.linear_cf_news1 = nn.Linear(opt.user_dim, 1)
        self.linear_cf_cate1 = nn.Linear(opt.cat_dim, 1)



    def forward(self, click_t, click_cat, click_sub, candi_t, candi_cat, candi_sub, ipw_candi, ipw_click):

        if self.opt.dt == 'large':
            cat_mask = torch.where(click_cat == 2, 1, 0) + torch.where(click_cat == 4, 1, 0) + torch.where(click_cat == 1, 1, 0)
        elif self.opt.dt == 'adressa':
            cat_mask = torch.where(click_cat == 5, 0, 1)

        click_news_pop, click_cat_pop = self.news_encoder(click_t*cat_mask.unsqueeze(-1), click_cat*cat_mask, click_sub*cat_mask)
        click_news_unpop, click_cat_unpop = self.news_encoder(click_t*(1-cat_mask).unsqueeze(-1), click_cat*(1-cat_mask), click_sub*(1-cat_mask))

        user_news_pop = self.user_encoder_pop(click_news_pop)       
        user_cat_pop = self.ucat_encoder_pop(click_cat_pop)

        click_news_unpop, click_cat_unpop = self.news_encoder(click_t*(1-cat_mask).unsqueeze(-1), click_cat*(1-cat_mask), click_sub*(1-cat_mask))
        user_news_unpop = self.user_encoder_unpop(click_news_unpop)
        user_cat_unpop = self.ucat_encoder_unpop(click_cat_unpop)
        cand_news, cand_cat = self.news_encoder(candi_t, candi_cat, candi_sub) 

        cf_loss, ipw_score = self.counterfactual(user_news_pop, user_cat_pop, cand_news, cand_cat, cat_mask, ipw_candi, ipw_click)
        final_score = self.compute_score(cand_news, cand_cat, user_news_pop, user_news_unpop, user_cat_pop, user_cat_unpop, ipw_score, False)

        return final_score, cf_loss

    def counterfactual(self, user_news_pop, user_cat_pop, cand_news, cand_cat, cat_mask, ipw_candi, ipw_click):
        cf_loss = None
        prosensity_score = None

        click_news2pop = self.linear_cf_news1(user_news_pop.detach())
        click_cate2pop = self.linear_cf_cate1(user_cat_pop.detach())
        cf_loss = (self.opt.alpha * click_news2pop + self.opt.beta * click_cate2pop).squeeze(-1)
        click_pro = torch.sum(ipw_click[:,:,0] * (1-cat_mask), -1) + 1
        imprs_pro = torch.sum(ipw_click[:,:,1] * (1-cat_mask), -1) + 1
        prosensity_score = torch.pow((imprs_pro.float() / click_pro) - 1, self.opt.gamma)

        return (cf_loss, prosensity_score)


    def encode_n(self, title, cat, sub_cat):
        return self.news_encoder(title, cat, sub_cat, training=0)

    def encode_u(self, news_fea_pop, news_fea_unpop):
        return self.user_encoder_pop(news_fea_pop), self.user_encoder_unpop(news_fea_unpop)

    def encode_c(self, news_fea_pop, news_fea_unpop):
        return self.ucat_encoder_pop(news_fea_pop), self.ucat_encoder_unpop(news_fea_unpop)

    def compute_score(self, cand_news, cand_cat, user_news_pop, user_news_unpop, user_cat_pop, user_cat_unpop, ipw_scores, infer=True):
        news_pop_score = cand_news.matmul(user_news_pop.unsqueeze(2))   # batch * cand * 1
        cat_pop_score = cand_cat.matmul(user_cat_pop.unsqueeze(2))

        news_unpop_score = cand_news.matmul(user_news_unpop.unsqueeze(2))
        cat_unpop_score = cand_cat.matmul(user_cat_unpop.unsqueeze(2))
        if infer == False:
            news_unpop_score *= ipw_scores.unsqueeze(-1).unsqueeze(-1)
            cat_unpop_score *= ipw_scores.unsqueeze(-1).unsqueeze(-1)
            pass
        all_score = torch.cat((news_pop_score, news_unpop_score, cat_pop_score, cat_unpop_score), -1)

        news_pop_weight = self.linear_news_pop(user_news_pop) # b * 1
        cat_pop_weight = self.linear_cat_pop(user_cat_pop)
        news_unpop_weight = self.linear_news_unpop(user_news_unpop)
        cat_unpop_weight = self.linear_cat_unpop(user_cat_unpop)

        all_weight = torch.cat((news_pop_weight, news_unpop_weight, cat_pop_weight, cat_unpop_weight), -1) # b * 4
        all_weight = F.softmax(all_weight, -1)

        pop_score = torch.sum(all_weight[:,0].unsqueeze(1) * news_pop_score + all_weight[:,2].unsqueeze(1) * cat_pop_score, -1)
        unp_score = torch.sum(all_weight[:,1].unsqueeze(1) * news_unpop_score + all_weight[:,3].unsqueeze(1) * cat_unpop_score, -1)

        final_score = torch.sum(all_weight.unsqueeze(1) * all_score, -1)

        if infer:
            debias = 0
            click_news2pop = self.linear_cf_news1(user_news_pop)
            click_cate2pop = self.linear_cf_cate1(user_cat_pop)
            ipw_score = torch.pow(((ipw_scores[:,:,1]+1).float() / (ipw_scores[:,:,0]+1)), self.opt.gamma)

            debias = - self.opt.alpha * click_news2pop - self.opt.alpha * click_cate2pop
            debias = debias * ipw_score
            final_score += debias
            pop_score += debias
            return final_score, pop_score, unp_score

        return final_score.squeeze(-1)



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