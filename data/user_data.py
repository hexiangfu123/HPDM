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


class UserData(Dataset):

    def __init__(self, mode, dt='large', test=False):
        self.dt = dt
        if mode == 'Dev':
            path = f'./data/{dt}/dev'
        if mode == 'Test':
            path = f'./data/{dt}/test'
        self.click_news_indexes = np.load(f"{path}/click_news_indexes.npy", allow_pickle=True)
        self.click_cat_indexes = np.load(f"{path}/click_cat_indexes.npy", allow_pickle=True)
        # self.news_mask_cl = np.load(f"./data/{dt}/train/news_2_popularity_cl.npy", allow_pickle=True)


    def __getitem__(self, idx):
        assert idx < len(self)
        click_nids = self.click_news_indexes[idx]
        cat_nids = self.click_cat_indexes[idx]
        if self.dt == 'large':
            cat_mask = np.where(cat_nids == 2, 1, 0) + np.where(cat_nids == 1, 1, 0) + np.where(cat_nids == 4, 1, 0)
        elif self.dt == 'adressa':
            cat_mask = np.where(cat_nids == 5, 0, 1)
        return click_nids, cat_mask, cat_mask

    def __len__(self):
        return len(self.click_news_indexes)
