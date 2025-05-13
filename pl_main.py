# -*- coding: utf-8 -*-

import time
import os
import random
import fire
import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as CKPT

from data import NewsData
from config import opt

from model import Model
import methods


def now():
    return str(time.strftime("%m-%d_%H:%M:%S"))


def collate_fn(batch):
    # label_list, candidate_news_indexs, click_news_indexes, user_indexes....
    data = zip(*batch)
    data = [torch.LongTensor(d) for d in data]
    return data


def train(**kwargs):

    opt.parse(kwargs)

    pl.seed_everything(opt.seed)
    print("loading npy data...")
    train_data = NewsData("Train", opt.dt)
    test_data = NewsData("Test", opt.dt)

    print(f"train data: {len(train_data)},dev data: {len(test_data)}")

    train_dataloader = DataLoader(train_data,
                                  opt.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_data,
                                 opt.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn, num_workers=opt.num_workers)

    Net = getattr(methods, opt.model)
    model = Model(Net, opt)

    ckpt = CKPT(filepath='./checkpoints/{epoch}-{group_auc:.4f}-{mean_mrr:.4f}-{ndcg@5:.4f}-{ndcg@10:.4f}')

    trainer = pl.Trainer(gpus=opt.gpu_ids, max_epochs=opt.epochs, distributed_backend='ddp', default_root_dir='./checkpoints/',
                         checkpoint_callback=ckpt, num_sanity_val_steps=0, fast_dev_run=False
                         )

    trainer.fit(model, train_dataloader, test_dataloader)
    print("*****************FINISHED***********************")
    print(f"{now()}: best score: {ckpt.best_model_score}, the path: {ckpt.best_model_path.replace(os.getcwd(), '')}")


if __name__ == "__main__":
    fire.Fire()
