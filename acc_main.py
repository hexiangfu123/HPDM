# -*- coding: utf-8 -*-

import time
import random
import fire
import pdb
import pickle

import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.news_data import NewsData
from data.text_data import TextData
from data.user_data import UserData
from config import opt
import models
from models.hpdm import HPDM
from tqdm import tqdm
import ipdb
from utils import group_resuts, group_resuts_test, cal_metric

from accelerate import Accelerator


accelerator = Accelerator()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def now():
    return str(time.strftime('%m-%d_%H:%M:%S'))


def collate_fn(batch):
    '''
    label_list, candidate_news_indexs, click_news_indexes, user_indexes
    '''
    data = zip(*batch)
    li = []
    for i, d in enumerate(data):
        li.append(torch.IntTensor(np.array(d, dtype='int32')))

    return li



def train(**kwargs):
    opt.parse(kwargs)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    model = HPDM(opt)

    accelerator.print("loading npy data...")
    train_data = NewsData("Train", opt.dt, enc=opt.enc)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, collate_fn=collate_fn)

    dev_data = NewsData("Dev", opt.dt)
    dev_dataloader = DataLoader(dev_data, 1, shuffle=False, collate_fn=collate_fn)
    accelerator.print(f'train data: {len(train_data)},dev data: {len(dev_data)}')

    news_data = TextData("Test", opt.dt, enc=opt.enc)
    accelerator.print(f'news data: {len(news_data)}')

    news_dataloader = DataLoader(news_data, opt.batch_size*16, shuffle=False, collate_fn=collate_fn)

    user_data = UserData("Dev", opt.dt)
    user_dataloader = DataLoader(user_data, opt.batch_size*16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), opt.lr)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)



    ctr = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    accelerator.print("start traning..")
    total_auc = 0.0
    opt.print_step = 160000
    for epoch in range(opt.epochs):
        total_loss = 0.0
        accelerator.print(f"{now()} Epoch {epoch}:")
        steps_in_epoch = len(train_dataloader)
        model.train()

        for idx, data in tqdm(enumerate(train_dataloader), total=steps_in_epoch, desc=f'Train epoch {epoch}'):
            click_t, click_cat, click_sub, click_nids, candi_t, candi_cat, candi_sub, candi_nids, label_list,  ipw_candi, ipw_click = data
            out, cf_pred = model(click_t, click_cat, click_sub, candi_t, candi_cat, candi_sub, ipw_candi, ipw_click)
            label_list = label_list.max(1)[1]
            nce_loss = ctr(out, label_list)
            loss = nce_loss
            click_score = torch.pow((torch.sum(ipw_click[:,:,1], -1) + 1).float() / (torch.sum(ipw_click[:,:,0])+1), opt.gamma)
            cf_loss = mse(cf_pred, click_score)
            loss += 0.1 * cf_loss

            optimizer.zero_grad()
            total_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            if idx % opt.print_step == opt.print_step - 1:
                accelerator.print(f"\t\t {now()} {idx} steps finised")
                res, auc = dev_test(model, dev_dataloader, news_dataloader, user_dataloader,
                    opt.metrics, opt.print_step, opt.pal)
                accelerator.print(f"\t the res in dev set: {res}")

        mean_loss = total_loss / len(train_data)
        accelerator.print(f"\t{now()}train loss: {mean_loss:.6f};")
        if epoch > -1:
            accelerator.print(f"\t{now()} start test...")
            res, auc = dev_test(model, dev_dataloader, news_dataloader, user_dataloader,
                                opt.metrics, opt.print_step, opt.pal)
            accelerator.wait_for_everyone()
            unwrappered_model = accelerator.unwrap_model(model)

            accelerator.print(f"\t the res in dev set: {res}")
            if total_auc < auc:
                accelerator.print(f"\t\t the best res updates from {total_auc} to {auc}")
                accelerator.save(unwrappered_model.state_dict(), f'./checkpoints/saved_model_{opt.mode}_{opt.enc}.pth')
        opt.print_step = opt.print_step / 2


def dev_test(model, dev_dataloader, news_dataloader, user_dataloader, metrics, print_step, pal):
    model.eval()

    labels = []
    preds = []

    with torch.no_grad():
        news_fea = []
        cat_fea = []
        for num, data in tqdm(enumerate(news_dataloader), total=len(news_dataloader), desc=f'News emb'):
            title, cat, sub_cat, news_mask, cat_mask = [i.to(accelerator.device) for i in data]

            news_emb, cat_emb = model.module.encode_n(title, cat, sub_cat)

            news_fea.append(news_emb)
            cat_fea.append(cat_emb)
        news_feas = torch.cat(news_fea, dim=0)
        cat_feas = torch.cat(cat_fea, dim=0)


        user_fea_pop = []
        user_fea_unp = []
        ucat_fea_pop = []
        ucat_fea_unp = []

        for step, data in tqdm(enumerate(user_dataloader), total=len(user_dataloader), desc=f'User emb'):
            click_nids, news_mask, cat_mask = [i.to(accelerator.device) for i in data]
            news_fea_pop = news_feas[click_nids.long() * cat_mask]
            news_fea_unpop = news_feas[click_nids.long() * (1-cat_mask)]
            cat_fea_pop = cat_feas[click_nids.long() * cat_mask]
            cat_fea_unpop = cat_feas[click_nids.long() * (1-cat_mask)]


            user_news_pop, user_news_unpop = model.module.encode_u(news_fea_pop, news_fea_unpop)
            user_cat_pop, user_cat_unpop = model.module.encode_c(cat_fea_pop, cat_fea_unpop)

            user_fea_pop.append(user_news_pop)
            user_fea_unp.append(user_news_unpop)
            ucat_fea_pop.append(user_cat_pop)
            ucat_fea_unp.append(user_cat_unpop)
            # user_fea.append(model.module.encode_u(news_fea_n))
        user_feas_pop = torch.cat(user_fea_pop, dim=0)
        user_feas_unp = torch.cat(user_fea_unp, dim=0)
        ucat_feas_pop = torch.cat(ucat_fea_pop, dim=0)
        ucat_feas_unp = torch.cat(ucat_fea_unp, dim=0)

        # calculate scoring
        AUC, MRR, nDCG5, nDCG10 = [], [], [], []
        for step, data in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), desc=f'Score cpt'):
            click_nids, candi_nids, label_list, ipw_candi, ipw_click = [i.to(accelerator.device) for i in data]

            candi_news = news_feas[candi_nids[0]].unsqueeze(0)
            candi_cat = cat_feas[candi_nids[0]].unsqueeze(0)
            user_new_pop = user_feas_pop[step].unsqueeze(0)
            user_new_unp = user_feas_unp[step].unsqueeze(0)
            user_cat_pop = ucat_feas_pop[step].unsqueeze(0)
            user_cat_unp = ucat_feas_unp[step].unsqueeze(0)

            score = model.module.compute_score(candi_news, candi_cat, user_new_pop, user_new_unp, user_cat_pop, user_cat_unp, ipw_candi)

            out = score.reshape(-1).tolist()
            label_list = label_list.reshape(-1).tolist()

            preds.append(out)
            labels.append(label_list)

        for lab, pre in zip(labels, preds):
            res = cal_metric(lab, pre, metrics)
            AUC.append(res['auc'])
            MRR.append(res['mean_mrr'])
            nDCG5.append(res['ndcg@5'])
            nDCG10.append(res['ndcg@10'])
        res_ = {'auc': np.array(AUC).mean(), 'mean_mrr': np.array(MRR).mean(),
                'ndcg@5': np.array(nDCG5).mean(), 'ndcg@10': np.array(nDCG10).mean()}
        str_res = [f"{k}:{v}" for k, v in res_.items()]
        # np.save(f"./results/{now()}_{res['auc']}_preds.npy", np.array(preds, dtype=object))

    return ' '.join(str_res), res_['auc']


def get_optimizer_params(weight_decay, model):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                             'weight_decay': weight_decay},
                            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0}]

        return optimizer_params


if __name__ == "__main__":
    fire.Fire()