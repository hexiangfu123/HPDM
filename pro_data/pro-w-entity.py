# -*- coding: utf-8 -*-

import pickle
import random
import re
import sys
import pdb
import json

import numpy as np

from hparams import opt
from transformers import AutoTokenizer

DATA_PATH = './raw/large'
SAVE_PATH = '../data'

Word_dict_PATH = './raw/word_dict.pkl'
User_dict_PATH = './raw/uid2index_llarge.pkl'
Cat_dict_PATH = './raw/category2index_llarge.pkl'
Sub_dict_PATH = './raw/subcategory2index_llarge.pkl'
Entity_dict_PATH = './raw/entityid2index_large.pkl'

pat = re.compile(r"[\w]+|[.,!?;|]")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class MINDIterator:
    def __init__(self, opt, NPRATIO=-1, t='glove', test=False):
        self.opt = opt
        self.NPRATIO = NPRATIO
        self.word_dict = self.load_dict(Word_dict_PATH)
        self.cat_dict = self.load_dict(Cat_dict_PATH)
        self.sub_dict = self.load_dict(Sub_dict_PATH)
        self.uid2index = self.load_dict(User_dict_PATH)
        self.eid2index = self.load_dict(Entity_dict_PATH)
        self.tokenizer = t
        self.test = test

    def newsample(self, news, ratio):
        if ratio > len(news):
            return news + [0] * (ratio - len(news))
        else:
            return random.sample(news, ratio)

    def wordtoken(self, sent):
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []

    def load_dict(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def init_news(self, news_file, title_size, abs_size, word_dict):
        print('\tinit news...')
        nid2index = {}
        news_title = ['']
        category = [0]
        sub = [0]
        entity = [[0, 0]]
        abs = ['']
        with open(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, entity_list, _ = line.strip("\n").split("\t")
                if nid in nid2index:
                    continue
                nid2index[nid] = len(nid2index) + 1
                title = self.wordtoken(title)
                ab = self.wordtoken(ab)
                news_title.append(title)
                category.append(self.cat_dict[vert])
                sub.append(self.sub_dict[subvert])
                abs.append(ab)
                entity_list = json.loads(entity_list)
                eid_list = []
                for ent in entity_list:
                    id_c = ent['WikidataId']
                    if id_c in self.eid2index.keys():
                        eid_list.append(self.eid2index[id_c])
                    else:
                        eid_list.append(0)
                if len(eid_list) < 2:
                    eid_list.extend([0] * (2 - len(eid_list)))
                else:
                    eid_list = eid_list[:2]
                entity.append(eid_list)

        def glove_tokenize(news_item, avg_size):
            news_title_index = np.zeros((len(news_item), avg_size),
                                        dtype=np.uint16)
            for news_index in range(len(news_item)):
                title = news_item[news_index]
                for word_index in range(min(avg_size, len(title))):
                    if title[word_index] in word_dict:
                        news_title_index[news_index, word_index] = word_dict[
                            title[word_index].lower()]
            return news_title_index

        def bert_tokenize():
            news = [' '.join(t) for t in news_title]
            input_ids = bert_tokenizer(news, max_length=title_size*2, padding=True, truncation=True)['input_ids']
            return np.array(input_ids, dtype=np.uint16)

        self.nid2index = nid2index
        if self.tokenizer == 'glove':
            self.news_title_index = glove_tokenize(news_title, title_size)
            self.abs = glove_tokenize(abs, abs_size)
        else:
            self.news_title_index = bert_tokenize()
            self.abs = bert_tokenize()
        self.category = category
        self.sub = sub
        self.entity = entity
        return nid2index, self.news_title_index

    def init_behaviors_test(self, behaviors_file, his_size):
        print('\tinit test_behaviors...')
        histories = []
        imprs = []
        impr_indexes = []
        uindexes = []
        his_cats = []
        his_subs = []
        his_entities = []

        with open(behaviors_file, 'r') as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip('\n').split('\t')[-4:]
                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (his_size - len(history)) + history[:his_size]
                his_cat = []
                his_sub = []
                his_entity = []
                for his in history:
                    his_cat.append(self.category[his])
                    his_sub.append(self.sub[his])
                    his_entity.append(self.entity[his])
                impr_news = [
                    self.nid2index[i] for i in impr.split()
                ]

                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                histories.append(history)
                his_cats.append(his_cat)
                his_subs.append(his_sub)
                his_entities.append(his_entity)
                imprs.append(impr_news)
                uindexes.append(uindex)
                impr_indexes.append(impr_index)
                impr_index += 1
        self.histories = histories
        self.his_cats = his_cats
        self.his_subs = his_subs
        self.his_entities = his_entities
        self.imprs = imprs
        self.uindexes = uindexes
        self.impr_indexes = impr_indexes
        return histories, imprs, uindexes, impr_indexes

    def init_behaviors(self, behaviors_file, his_size):
        print('\tinit behaviors...')
        histories = []
        imprs = []
        labels = []
        impr_indexes = []
        uindexes = []
        his_cats = []
        his_subs = []
        his_entities = []

        with open(behaviors_file, 'r') as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip('\n').split('\t')[-4:]
                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (his_size - len(history)) + history[:his_size]
                his_cat = []
                his_sub = []
                his_entity = []
                for his in history:
                    his_cat.append(self.category[his])
                    his_sub.append(self.sub[his])
                    his_entity.append(self.entity[his])
                impr_news = [
                    self.nid2index[i.split('-')[0]] for i in impr.split()
                ]

                label = [int(i.split('-')[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                histories.append(history)
                his_cats.append(his_cat)
                his_subs.append(his_sub)
                his_entities.append(his_entity)
                imprs.append(impr_news)
                labels.append(label)
                uindexes.append(uindex)
                impr_indexes.append(impr_index)
                impr_index += 1
        self.histories = histories
        self.his_cats = his_cats
        self.his_subs = his_subs
        self.imprs = imprs
        self.labels = labels
        self.uindexes = uindexes
        self.impr_indexes = impr_indexes
        self.his_entities = his_entities
        return histories, imprs, labels, uindexes, impr_indexes

    def parse_one_line_test(self, line):
        if self.NPRATIO > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [1] + [0] * self.NPRATIO

                n = self.newsample(negs, self.NPRATIO)
                candidate_title_index = [p] + n
                candidate_cat_index = []
                candidate_sub_index = []
                candidate_ent_index = []
                for i in [p]+n:
                    candidate_cat_index.append(self.category[i])
                    candidate_sub_index.append(self.sub[i])
                    candidate_ent_index.append(self.entity[i])
                click_title_index = self.histories[line]
                click_cat_index = self.his_cats[line]
                click_sub_index = self.his_subs[line]
                click_ent_index = self.his_entities[line]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_cat_index,
                    candidate_sub_index,
                    candidate_ent_index,
                    click_cat_index,
                    click_sub_index,
                    click_ent_index,
                )
        else:
            impr = self.imprs[line]
            for news in impr:
                candidate_title_index = []
                impr_index = []
                user_index = []

                candidate_title_index.append(news)
                candidate_cat_index = self.category[news]
                candidate_sub_index = self.sub[news]
                candidate_ent_index = self.entity[news]

                click_title_index = self.histories[line]
                click_cat_index = self.his_cats[line]
                click_sub_index = self.his_subs[line]
                click_ent_index = self.his_entities[line]

                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_cat_index,
                    candidate_sub_index,
                    candidate_ent_index,
                    click_cat_index,
                    click_sub_index,
                    click_ent_index
                )

    def parse_one_line(self, line):
        if self.NPRATIO > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [1] + [0] * self.NPRATIO

                n = self.newsample(negs, self.NPRATIO)
                candidate_title_index = [p] + n
                candidate_cat_index = []
                candidate_sub_index = []
                candidate_ent_index = []
                for i in [p]+n:
                    candidate_cat_index.append(self.category[i])
                    candidate_sub_index.append(self.sub[i])
                    candidate_ent_index.append(self.entity[i])
                click_title_index = self.histories[line]
                click_cat_index = self.his_cats[line]
                click_sub_index = self.his_subs[line]
                click_ent_index = self.his_entities[line]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_cat_index,
                    candidate_sub_index,
                    candidate_ent_index,
                    click_cat_index,
                    click_sub_index,
                    click_ent_index,
                )
        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_title_index.append(news)
                candidate_cat_index = self.category[news]
                candidate_sub_index = self.sub[news]
                candidate_ent_index = self.entity[news]

                click_title_index = self.histories[line]
                click_cat_index = self.his_cats[line]
                click_sub_index = self.his_subs[line]
                click_ent_index = self.his_entities[line]

                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_cat_index,
                    candidate_sub_index,
                    candidate_ent_index,
                    click_cat_index,
                    click_sub_index,
                    click_ent_index,
                )

    def load_data_from_file(self, news_file, behavior_file, mode='Train'):
        if self.test:
            imp_indexes = []
            user_indexes = []
            candidate_news_indexes = []
            click_news_indexes = []
            candidate_cat_indexes = []
            candidate_sub_indexes = []
            candidate_ent_indexes = []
            click_cat_indexes = []
            click_sub_indexes = []
            click_ent_indexes = []
            self.init_news(news_file, self.opt.title_size, self.opt.abs_size, self.word_dict)
            self.init_behaviors_test(behavior_file, self.opt.his_size)

            for i in range(len(self.imprs)):
                for imp_index, user_index, candidate_news_index, click_news_index, candidate_cat_index, candidate_sub_index, candidate_ent_index, click_cat_index, click_sub_index, click_ent_index in self.parse_one_line_test(i):
                    candidate_news_indexes.append(candidate_news_index)
                    click_news_indexes.append(click_news_index)
                    imp_indexes.append(imp_index)
                    user_indexes.append(user_index)
                    candidate_cat_indexes.append(candidate_cat_index)
                    candidate_sub_indexes.append(candidate_sub_index)
                    candidate_ent_indexes.append(candidate_ent_index)
                    click_cat_indexes.append(click_cat_index)
                    click_sub_indexes.append(click_sub_index)
                    click_ent_indexes.append(click_ent_index)
            path = f'{SAVE_PATH}/test'
            print('\t start saving...')
            np.save(f"{path}/imp_indexes.npy", imp_indexes)
            np.save(f"{path}/user_indexes.npy", user_indexes)
            np.save(f"{path}/candidate_news_indexes.npy", np.array(candidate_news_indexes, dtype=np.int32))
            np.save(f"{path}/click_news_indexes.npy", np.array(click_news_indexes, dtype=np.int32))
            np.save(f"{path}/news_title_index.npy", self.news_title_index)
            np.save(f"{path}/candidate_cat_indexes.npy", np.array(candidate_cat_indexes, dtype=np.int16))
            np.save(f"{path}/candidate_sub_indexes.npy", np.array(candidate_sub_indexes, dtype=np.int16))
            np.save(f"{path}/candidate_ent_indexes.npy", np.array(candidate_ent_indexes, dtype=np.int16))
            np.save(f"{path}/click_cat_indexes.npy", np.array(click_cat_indexes, dtype=np.int16))
            np.save(f"{path}/click_sub_indexes.npy", np.array(click_sub_indexes, dtype=np.int16))
            np.save(f"{path}/click_ent_indexes.npy", np.array(click_ent_indexes, dtype=np.int16))
            np.save(f"{path}/abs.npy", self.abs)

            return user_indexes
        else:
            label_list = []
            imp_indexes = []
            user_indexes = []
            candidate_news_indexes = []
            click_news_indexes = []
            candidate_cat_indexes = []
            candidate_sub_indexes = []
            candidate_ent_indexes = []
            click_cat_indexes = []
            click_sub_indexes = []
            click_ent_indexes = []

            self.init_news(news_file, self.opt.title_size, self.opt.abs_size, self.word_dict)
            self.init_behaviors(behavior_file, self.opt.his_size)

            for i in range(len(self.imprs)):
                for label, imp_index, user_index, candidate_news_index, click_news_index, candidate_cat_index, candidate_sub_index, candidate_ent_index, click_cat_index, click_sub_index, click_ent_index in self.parse_one_line(i):
                    candidate_news_indexes.append(candidate_news_index)
                    click_news_indexes.append(click_news_index)
                    imp_indexes.append(imp_index)
                    user_indexes.append(user_index)
                    label_list.append(label)
                    candidate_cat_indexes.append(candidate_cat_index)
                    candidate_sub_indexes.append(candidate_sub_index)
                    candidate_ent_indexes.append(candidate_ent_index)
                    click_cat_indexes.append(click_cat_index)
                    click_sub_indexes.append(click_sub_index)
                    click_ent_indexes.append(click_ent_index)
            if mode == 'Train':
                path = f'{SAVE_PATH}/train'
            elif mode == 'Dev':
                path = f'{SAVE_PATH}/dev'

            print('\t start saving...')
            np.save(f"{path}/label_list.npy", label_list)
            np.save(f"{path}/imp_indexes.npy", imp_indexes)
            np.save(f"{path}/user_indexes.npy", user_indexes)
            np.save(f"{path}/candidate_news_indexes.npy", np.array(candidate_news_indexes, dtype=np.int32))
            np.save(f"{path}/click_news_indexes.npy", np.array(click_news_indexes, dtype=np.int32))
            np.save(f"{path}/news_title_index.npy", self.news_title_index)
            np.save(f"{path}/candidate_cat_indexes.npy", np.array(candidate_cat_indexes, dtype=np.int16))
            np.save(f"{path}/candidate_sub_indexes.npy", np.array(candidate_sub_indexes, dtype=np.int16))
            np.save(f"{path}/click_cat_indexes.npy", np.array(click_cat_indexes, dtype=np.int16))
            np.save(f"{path}/click_sub_indexes.npy", np.array(click_sub_indexes, dtype=np.int16))
            np.save(f"{path}/abs.npy", self.abs)
            np.save(f"{path}/candidate_ent_indexes.npy", np.array(candidate_ent_indexes, dtype=np.int32))
            np.save(f"{path}/click_ent_indexes.npy", np.array(click_ent_indexes, dtype=np.int32))


            return user_indexes


if __name__ == "__main__":
    assert len(sys.argv) == 3, 'please specify the data type and glove/bert: \n\t python3 data_pro.py large glove'
    data_type = sys.argv[1]
    token_type = sys.argv[2]
    print(f"dataset: {data_type}; use {token_type} as tokenizer")

    assert data_type in ['large', 'small', 'llarge']
    assert token_type in ['glove', 'bert']

    if data_type == 'large':
        DATA_PATH = './raw/large'
        SAVE_PATH = '../data/large'
        User_dict_PATH = './raw/uid2index_llarge.pkl'
        Cat_dict_PATH = './raw/category2index_llarge.pkl'
        Sub_dict_PATH = './raw/subcategory2index_llarge.pkl'
        Entity_dict_PATH = './raw/entityid2index_large.pkl'
    elif data_type == 'small':
        DATA_PATH = './raw/small'
        SAVE_PATH = '../data/small'
        User_dict_PATH = './raw/uid2index_small.pkl'
        Cat_dict_PATH = './raw/category2index_small.pkl'
        Sub_dict_PATH = './raw/subcategory2index_small.pkl'
    else:
        DATA_PATH = './raw/llarge'
        SAVE_PATH = '../data/llarge'
        User_dict_PATH = './raw/uid2index_llarge.pkl'
        Cat_dict_PATH = './raw/category2index_llarge.pkl'
        Sub_dict_PATH = './raw/subcategory2index_llarge.pkl'

    train_iterator = MINDIterator(opt, NPRATIO=opt.ratio_K, t=token_type)
    print("process train data...")
    train_users = train_iterator.load_data_from_file(
        f"{DATA_PATH}/train/news.tsv", f"{DATA_PATH}/train/behaviors.tsv", 'Train')

    dev_iterator = MINDIterator(opt, t=token_type)
    print("process dev data...")
    dev_users = dev_iterator.load_data_from_file(
        f"{DATA_PATH}/dev/news.tsv", f"{DATA_PATH}/dev/behaviors.tsv", 'Dev')

    if data_type == 'large':
        test_iterator = MINDIterator(opt, t=token_type, test=True)
        print("process test data...")
        test_users = test_iterator.load_data_from_file(
            f"{DATA_PATH}/test/news.tsv", f"{DATA_PATH}/test/behaviors.tsv", 'Test')
