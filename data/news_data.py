# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
import pickle
import ipdb
'''
自定义的数据集类需要继承torch.utils.data.Dataset这个抽象类，并且要实现两个函数：__getitem__与__len__
__getitem___是根据索引返回一个数据以及其标签
__len__返回一个Sample的大小
'''


class NewsData(Dataset):

    def __init__(self, mode, dt='large', test=False, norm='max', pop_num=-1, tokenizer='roberta'):
        self.mode = mode
        if mode == 'Train':
            path = f'./data/{dt}/train'
            
        if mode == 'Dev':
            path = f'./data/{dt}/dev'
        if mode == 'Test':
            path = f'./data/{dt}/test'
        pop_path = f'./data/{dt}/train'
        self.norm = norm
        self.pop_num = pop_num
        self.test = test
        self.dt = dt
        self.pro = 0.2
        self.tokenizer = tokenizer
        # ipdb.set_trace()

        self.news_title_index = np.load(f"{path}/news_title_index_{self.tokenizer}.npy", allow_pickle=True)
        # self.news_sapo_index = np.load(f"{path}/news_sapo_index_{self.tokenizer}.npy", allow_pickle=True)
        self.click_cat_indexes = np.load(f"{path}/click_cat_indexes.npy", allow_pickle=True)
        self.click_sub_indexes = np.load(f"{path}/click_sub_indexes.npy", allow_pickle=True)
        self.click_news_indexes = np.load(f"{path}/click_news_indexes.npy", allow_pickle=True)
        self.user_indexes = np.load(f"{path}/uindexes_list.npy", allow_pickle=True)


        self.news_popularity = np.load(f"{pop_path}/news_popularity.npy", allow_pickle=True)
        self.click_stages = np.load(f"{pop_path}/click_stages.npy", allow_pickle=True)
        self.news_popularity = np.cumsum(self.news_popularity, 1)

        self.max_popularity = (np.max(self.news_popularity, axis=0) + 1) / self.pop_num
        popularity = self.news_popularity / self.max_popularity
        self.news_popularity_class = popularity.astype('int')
    
        self.max_popularity = (np.max(self.news_popularity, axis=0) + 1)
        self.news_popularity_number = self.news_popularity + 1
        self.news_popularity_number = (self.news_popularity_number) / (self.max_popularity)
        # ipdb.set_trace()
        if mode == 'Train':
            
            self.candidate_news_indexs = np.load(f"{path}/candidate_news_indexes.npy", allow_pickle=True)
            self.candidate_cat_indexes = np.load(f"{path}/candidate_cat_indexes.npy", allow_pickle=True)
            self.candidate_sub_indexes = np.load(f"{path}/candidate_sub_indexes.npy", allow_pickle=True)

            self.label_list = np.load(f"{path}/label_list.npy", allow_pickle=True)

        else:
            with open(f"{path}/candidate_news_indexes.pkl", 'rb') as f:
                self.candidate_news_indexs = pickle.load(f)
            if mode == 'Dev':
                # ipdb.set_trace()
                with open(f"{path}/candidate_cat_indexes.pkl", 'rb') as f:
                    self.candidate_cat_indexes = pickle.load(f)
                with open(f"{path}/candidate_sub_indexes.pkl", 'rb') as f:
                    self.candidate_sub_indexes = pickle.load(f)
                with open(f"{path}/label_list.pkl", 'rb') as f:
                    self.label_list = pickle.load(f)

    def __getitem__(self, idx):
        assert idx < len(self)

        click_nids = self.click_news_indexes[idx]
        candidate_nids = self.candidate_news_indexs[idx]
        # uid = self.user_indexes[idx]
        # ipdb.set_trace()
        if self.mode == 'Test':
            return [click_nids, candidate_nids]
        if self.mode == 'Dev':

            label_list = self.label_list[idx]
            candidate_cat_index = self.candidate_cat_indexes[idx]
            click_cat_index = self.click_cat_indexes[idx]
            stages = self.click_stages[idx]
            candi_popularity = self.news_popularity_class[candidate_nids]
            click_popularity = self.news_popularity_number[candidate_nids]
            candi_pop_class =  candi_popularity[:, stages]
            cand_pop_number = click_popularity[:, stages]

            user_his_len = np.count_nonzero(click_nids) 
            if user_his_len == 0:
                user_his_len = 50
            user_pop_sum = self.news_popularity_number[click_nids][:, 0]
            user_pop_avg = np.sum(user_pop_sum[50-user_his_len:])/user_his_len
            
            user_comformity = np.power(cand_pop_number*user_pop_avg, 0.001)

            return [click_nids, candidate_nids, candidate_cat_index, click_cat_index, label_list, candi_pop_class, user_comformity]
        else:
            # click news info
            click_title_indexes = self.news_title_index[click_nids]
            # click_sapo_indexes = self.news_sapo_index[click_nids]
            click_cat_index = self.click_cat_indexes[idx]
            # candiate news info 
            label_list = self.label_list[idx]
            candidate_title_indexs = self.news_title_index[candidate_nids]
            # candidate_sapo_indexs = self.news_sapo_index[candidate_nids]
            candidate_cat_index = self.candidate_cat_indexes[idx]
            candidate_sub_index = self.candidate_sub_indexes[idx]
            # popularity info
            stages = self.click_stages[idx]
            candi_popularity = self.news_popularity_class[candidate_nids]
            click_popularity = self.news_popularity_number[candidate_nids]
            candi_pop_class =  candi_popularity[:, stages]
            cand_pop_number = click_popularity[:, stages]
            
            user_his_len = np.count_nonzero(click_nids) 
            if user_his_len == 0:
                user_his_len = 50
            user_pop_sum = self.news_popularity_number[click_nids][:, 0]
            user_pop_avg = np.sum(user_pop_sum[50-user_his_len:])/user_his_len

            user_comformity = np.power(cand_pop_number*user_pop_avg, 0.001)
            return [click_title_indexes, click_cat_index, click_nids, candidate_title_indexs, candidate_cat_index,
                    candidate_nids, label_list, candi_pop_class, user_comformity]

    def __len__(self):
        if not self.test:
            return len(self.label_list)
        else:
            return len(self.click_news_indexes)
