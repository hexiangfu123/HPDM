# -*- coding: utf-8 -*-
import time
import random
import fire
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaConfig, BertConfig, AutoConfig
from data.news_data import NewsData
from data.text_data import TextData
from data.user_data import UserData
from config import opt
from models.news_encoder import NewsEncoder
from models.hpdm import HPDM
from utils import group_resuts, group_resuts_test, cal_metric
from tqdm import tqdm
from accelerate import Accelerator


accelerator = Accelerator()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def now():
    return str(time.strftime('%m-%d_%H:%M:%S'))

class popularity_matrix():
    def __init__(self, news_pop_class, news_pop_number):
        # self.news_popularity_matrix = torch.FloatTensor(np.array(news_pop, dtype='float32')).to(accelerator.device)
        self.news_popularity_class_matrix = torch.IntTensor(np.array(news_pop_class, dtype='int32')).to(accelerator.device)
        self.news_popularity_number_matrix = torch.FloatTensor(np.array(news_pop_number, dtype='float32')).to(accelerator.device)
        #  self.cat_popularity_matrix = torch.IntTensor(np.array(cat_pop, dtype='int32')).to(accelerator.device)

    def get_popularity(self, click_nids):
        return [self.news_popularity_class_matrix[click_nids,-1],  self.news_popularity_number_matrix[click_nids,-1]]
    # def get_catpopularity(self, cat_nids):
    #     return self.cat_popularity_matrix[cat_nids, -1]
    # def get_popularity(self, candi_nids, click_nids):
    #     return self.news_popularity_matrix[candi_nids[0],-1], self.news_popularity_matrix[click_nids[0],-1]


def collate_fn(batch):
    '''
    label_list, candidate_news_indexs, click_news_indexes, user_indexes
    '''
    data = zip(*batch)
    li = []
    final = len(batch[0])-1
    for i, d in enumerate(data):
        if i < final:
            li.append(torch.IntTensor(np.array(d, dtype='int32')))
        else:
            li.append(torch.FloatTensor(np.array(d, dtype='float32')))

    return li


def train(**kwopt):
    opt.parse(kwopt)
    # ipdb.set_trace()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    accelerator.print("loading npy data...")
    train_data = NewsData("Train", opt.dt, norm=opt.norm, pop_num=opt.pop_size, tokenizer=opt.tokenizer)
    train_dataloader = DataLoader(train_data, opt.train_batch_size, shuffle=True, collate_fn=collate_fn)
    
    opt.user_num = max(train_data.user_indexes) + 1
    
    dev_data = NewsData("Dev", opt.dt, tokenizer=opt.tokenizer)
    dev_dataloader = DataLoader(dev_data, 1, shuffle=False, collate_fn=collate_fn)
    accelerator.print(f'train data: {len(train_data)},dev data: {len(dev_data)}')

    news_data = TextData("Test", opt.dt, tokenizer=opt.tokenizer)
    accelerator.print(f'news data: {len(news_data)}')
    # news_dataloader = DataLoader(news_data, opt.batch_size*2, shuffle=False, collate_fn=collate_fn)
    news_dataloader = DataLoader(news_data, opt.batch_size*4, shuffle=False, collate_fn=collate_fn)
    
    
    user_data = UserData("Dev", opt.dt)
    user_dataloader = DataLoader(user_data, 1, shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_tokenizer)
    if opt.tokenizer == 'bert':
        config = BertConfig.from_pretrained(opt.pretrained_embedding)
    elif opt.tokenizer == 'roberta':
        config = RobertaConfig.from_pretrained(opt.pretrained_embedding)
    elif opt.tokenizer == 'nbbert':
        config = AutoConfig.from_pretrained(opt.pretrained_embedding)
    
    news_encoder = NewsEncoder(path=opt.pretrained_embedding, config=config, apply_reduce_dim=opt.apply_reduce_dim, use_sapo=opt.use_sapo,
                                    dropout=opt.dropout, freeze_transformer=opt.freeze_transformer,
                                    word_embed_dim=opt.word_embed_dim, combine_type=opt.combine_type,
                                    lstm_num_layers=opt.lstm_num_layers, lstm_dropout=opt.lstm_dropout)

    if opt.model == 'HPDM':
        model = HPDM(news_encoder=news_encoder, use_category_bias=opt.use_category_bias,
                        num_context_codes=opt.num_context_codes, context_code_dim=opt.context_code_dim,
                        score_type=opt.score_type, dropout=opt.dropout, num_category=20,
                        category_embed_dim=opt.category_embed_dim, category_pad_token_id=0,
                        category_embed=opt.category_embed, pop_size=opt.pop_size)

    optimizer_params = get_optimizer_params(opt.weight_decay, model)
    optimizer = AdamW(optimizer_params, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    

    ctr = nn.CrossEntropyLoss(reduction='mean')
    accelerator.print("start traning..")
    matrix = popularity_matrix(train_data.news_popularity_class, train_data.news_popularity_number)
    

    epoch_auc = 0.0

    for epoch in range(opt.epochs):
        total_loss = 0.0
        accelerator.print(f"{now()} Epoch {epoch}:")
        model.train()
        steps_in_epoch = len(train_dataloader)
        
        accumulation_factor = (opt.gradient_accumulation_steps
                                   if steps_in_epoch > opt.gradient_accumulation_steps else steps_in_epoch)
        global_step = 0
        
        for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Train epoch {epoch}', position=0):

            click_t, click_cat, click_nids, candi_t, candi_cat, candi_nids, label_list, candi_pop, cat_pop = data
            
            click_t_mask = (click_t != tokenizer.pad_token_id)
            candi_t_mask = (candi_t != tokenizer.pad_token_id)
            click_nids_mask = (click_nids != 0)
            out, cf = model(title=candi_t, title_mask=candi_t_mask, his_title=click_t, category=candi_cat,
                            his_title_mask=click_t_mask, his_mask=click_nids_mask, sapo=None,
                            sapo_mask=None, his_sapo = None, his_category=click_cat, popularity=[candi_pop,cat_pop])


            target = label_list.argmax(dim=1)
            nce_loss = ctr(out, target)
            con_loss = ctr(cf, target)
            if opt.decorrelation:
                loss = nce_loss + 0.1 * con_loss

            optimizer.zero_grad()
            total_loss += loss.item() * click_t.size()[0]
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            global_step += 1
        
        mean_loss = total_loss / len(train_data)
        accelerator.print(f"\t{now()}train loss: {mean_loss:.6f};")
        if epoch > -1:
            accelerator.print(f"\t{now()} start test...")
            res, auc = dev_test1_parr(model, dev_dataloader, news_dataloader, user_dataloader,
                                    opt.metrics, pop_matrix=matrix, tokenizer=tokenizer)
            accelerator.wait_for_everyone()
            unwrappered_model = accelerator.unwrap_model(model)
            accelerator.print(f"\t the res in dev set: {res}")
            if auc > epoch_auc:
                accelerator.print(f"\t \t Best AUC score updates from {epoch_auc} to {auc}")
                accelerator.save(unwrappered_model.state_dict(), f'./checkpoints/{opt.model}_small_{opt.dt}_{res}.pth')
                epoch_auc = auc 


def dev_test1_parr(model, dev_dataloader, news_dataloader, user_dataloader, metrics, pop_matrix=None, tokenizer=None):
    model.eval()

    labels = []
    preds = []
    # pop_matrix = torch.IntTensor(np.array(pop_matrix, dtype='int32')).to(accelerator.device)
    with torch.no_grad():
        accelerator.print(f"\t{now()} News embedding Begin!")
        news_fea = []
        for num, data in tqdm(enumerate(news_dataloader), total=len(news_dataloader), desc='news repr load', position=0):
            title, cat, sub_cat, sapo = [i.to(accelerator.device) for i in data]
            # ipdb.set_trace()
            title_mask = (title != tokenizer.pad_token_id)
            sapo_mask = (sapo != tokenizer.pad_token_id)
            # ipdb.set_trace()
            news_repr = model.news_repr(title.unsqueeze(1), title_mask.unsqueeze(1), sapo.unsqueeze(1), sapo_mask.unsqueeze(1))
            news_fea.append(news_repr.squeeze(1))
            # news_fea.append(model.module.encode_n(title, cat, sub_cat))
        # accelerator.wait_for_everyone()
        news_feas = torch.cat(news_fea, dim=0)
        accelerator.print(f"\t{now()} News embedding done   {news_feas.size(0)}")

        # calculate scoring
        AUC, MRR, nDCG5, nDCG10 = [], [], [], []
        for step, data in tqdm(enumerate(dev_dataloader),total=len(dev_dataloader), desc='test', position=0):
            # ipdb.set_trace()
            click_nids, candi_nids, candidate_cat_index, click_cat_index, label_list, pop1, pop2 = [i.to(accelerator.device) for i in data]
            his_mask = (click_nids != 0)
            candi_news_fea = news_feas[candi_nids]
            click_news_fea = news_feas[click_nids]
            # ipdb.set_trace()
            out = model.compute_score(his_category=click_cat_index, category=candidate_cat_index, history_repr=click_news_fea,\
                                         candi_repr=candi_news_fea, his_mask=his_mask, popularity=[pop1, pop2])
            out = out.view(-1).tolist()
            label_list = label_list.reshape(-1).tolist()
            assert len(out) == len(label_list)
            # ipdb.set_trace()
            preds.append(out)
            labels.append(label_list)
        # accelerator.wait_for_everyone()
        for lab, pre in zip(labels, preds):
            # ipdb.set_trace()
            res = cal_metric(lab, pre, metrics)
            AUC.append(res['auc'])
            MRR.append(res['mean_mrr'])
            nDCG5.append(res['ndcg@5'])
            nDCG10.append(res['ndcg@10'])
        res_ = {'auc': np.array(AUC).mean(), 'mean_mrr': np.array(MRR).mean(),
                'ndcg@5': np.array(nDCG5).mean(), 'ndcg@10': np.array(nDCG10).mean()}
        str_res = [f"{k}:{v}" for k, v in res_.items()]
        # np.save(f"./results/{now()}_{res['auc']}_preds.npy", np.array(preds, dtype=object))
    # ipdb.set_trace()
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
