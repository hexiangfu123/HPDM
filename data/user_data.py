# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
import pickle

'''
自定义的数据集类需要继承torch.utils.data.Dataset这个抽象类，并且要实现两个函数：__getitem__与__len__
__getitem___是根据索引返回一个数据以及其标签
__len__返回一个Sample的大小
'''


class UserData(Dataset):

    def __init__(self, mode, dt='large', test=False):
        if mode == 'Dev':
            path = f'./data/{dt}/dev'
        if mode == 'Test':
            path = f'./data/{dt}/test'
        self.click_news_indexes = np.load(f"{path}/click_news_indexes.npy", allow_pickle=True)
        self.click_cat_indexes = np.load(f"{path}/click_cat_indexes.npy", allow_pickle=True)
        with open(f"{path}/candidate_cat_indexes.pkl", 'rb') as f:
                    self.candidate_cat_indexes = pickle.load(f)

    def __getitem__(self, idx):
        assert idx < len(self)

        click_nids = self.click_news_indexes[idx]
        click_cat_nids = np.array(self.click_cat_indexes[idx], dtype='int32')
        # ipdb.set_trace()
        # click_cat_nids.dtype = np.int32
        candi_cat_nids = np.array(self.candidate_cat_indexes[idx], dtype='int32')
        # 
        
        return [click_nids, click_cat_nids, candi_cat_nids]

    def __len__(self):
        return len(self.click_news_indexes)
