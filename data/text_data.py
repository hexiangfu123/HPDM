# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np

'''
自定义的数据集类需要继承torch.utils.data.Dataset这个抽象类，并且要实现两个函数：__getitem__与__len__
__getitem___是根据索引返回一个数据以及其标签
__len__返回一个Sample的大小
'''


class TextData(Dataset):

    def __init__(self, mode, dt='large', test=False, tokenizer='roberta'):

        path = f'./data/{dt}/train'
        self.test = test
        self.dt = dt
        self.tokenizer = tokenizer

        self.news_title_index = np.load(f"{path}/news_title_index_{self.tokenizer}.npy", allow_pickle=True)
        # self.news_sapo_index = np.load(f"{path}/news_sapo_index_{self.tokenizer}.npy", allow_pickle=True)
        self.news_cat_index = np.load(f"{path}/news_cat_index.npy", allow_pickle=True)
        
        self.news_sub_index = np.load(f"{path}/news_sub_index.npy", allow_pickle=True)

    def __getitem__(self, idx):
        assert idx < len(self)

        title = self.news_title_index[idx]
        cat = self.news_cat_index[idx]
        sub = self.news_sub_index[idx]
        sapo = title # self.news_sapo_index[idx]
        return [title, cat, sub, sapo]

    def __len__(self):
        return len(self.news_title_index)
