# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
import pickle
import ipdb
import scipy.stats as stats
'''
自定义的数据集类需要继承torch.utils.data.Dataset这个抽象类，并且要实现两个函数：__getitem__与__len__
__getitem___是根据索引返回一个数据以及其标签
__len__返回一个Sample的大小
'''


class NewsData(Dataset):

    def __init__(self, mode, dt='large', test=False, enc='glove', pop_num=-1):
        self.mode = mode
        if mode == 'Train':
            path = f'./data/{dt}/train'

        if mode == 'Dev':
            path = f'./data/{dt}/dev'
        if mode == 'Test':
            path = f'./data/{dt}/test'
        self.enc = enc
        self.pop_num = pop_num
        self.test = test
        self.dt = dt
        pop_path = f'./data/{dt}/train'



        self.news_title_index = np.load(f"{path}/news_title_index_{self.enc}.npy", allow_pickle=True)
        self.click_cat_indexes = np.load(f"{path}/click_cat_indexes.npy", allow_pickle=True)
        self.click_sub_indexes = np.load(f"{path}/click_sub_indexes.npy", allow_pickle=True)
        self.click_news_indexes = np.load(f"{path}/click_news_indexes.npy", allow_pickle=True)
        self.user_indexes = np.load(f"{path}/uindexes_list.npy", allow_pickle=True)
        self.news_popularity = np.load(f"{pop_path}/news_popularity.npy", allow_pickle=True)

        if self.dt == 'adressa':
            ipw = np.sum(self.news_popularity, axis=1)
            all_score = np.tile(np.max(ipw), len(ipw))
            self.ipw_score = np.stack((ipw, all_score), 1)
        else:
            self.ipw_score = np.load(f"{pop_path}/ipw_score.npy", allow_pickle=True)
        if mode == 'Train':

            self.candidate_news_indexs = np.load(f"{path}/candidate_news_indexes.npy", allow_pickle=True)
            self.candidate_cat_indexes = np.load(f"{path}/candidate_cat_indexes.npy", allow_pickle=True)
            self.candidate_sub_indexes = np.load(f"{path}/candidate_sub_indexes.npy", allow_pickle=True)

            self.label_list = np.load(f"{path}/label_list.npy", allow_pickle=True)

        else:
            with open(f"{path}/candidate_news_indexes.pkl", 'rb') as f:
                self.candidate_news_indexs = pickle.load(f)
            if mode == 'Dev':

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

        if self.mode == 'Test':
            return [click_nids, candidate_nids]
        if self.mode == 'Dev':

            label_list = self.label_list[idx]
            candidate_cat_index = self.candidate_cat_indexes[idx]
            click_cat_index = self.click_cat_indexes[idx]
            news_mask_clk = self.ipw_score[click_nids]
            news_mask_can = self.ipw_score[candidate_nids]
            return [click_nids, candidate_nids, label_list, news_mask_can, news_mask_clk]
        else:
            click_title_indexes = self.news_title_index[click_nids]
            click_cat_index = self.click_cat_indexes[idx]
            click_sub_index = self.click_sub_indexes[idx]

            label_list = self.label_list[idx]
            candidate_title_indexs = self.news_title_index[candidate_nids]

            candidate_cat_index = self.candidate_cat_indexes[idx]
            candidate_sub_index = self.candidate_sub_indexes[idx]
            news_mask_clk = self.ipw_score[click_nids]
            news_mask_can = self.ipw_score[candidate_nids]
            return [click_title_indexes, click_cat_index, click_sub_index, click_nids,
                    candidate_title_indexs, candidate_cat_index, candidate_sub_index,
                    candidate_nids, label_list, news_mask_can, news_mask_clk]

    def __len__(self):
        if not self.test:
            return len(self.label_list)
        else:
            return len(self.click_news_indexes)