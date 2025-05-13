# -*- coding: utf-8 -*-

import pickle
import random
import re
import sys
import pdb
import json
import ipdb
import numpy as np

from hparams import opt
from transformers import AutoTokenizer
import pdb

DATA_PATH = './raw/small'
SAVE_PATH = '../data'

Word_dict_PATH = './raw/word_dict.pkl'
User_dict_PATH = './raw/uid2index_llarge.pkl'
Cat_dict_PATH = './raw/category2index_llarge.pkl'
Sub_dict_PATH = './raw/subcategory2index_llarge.pkl'
Emb_dict_PATH = './raw/embedding_small.npy'
token_type = 'glove'
pat = re.compile(r"[\w]+|[.,!?;|]")
bert_tokenizer = AutoTokenizer.from_pretrained("../../../pre-train-model/bert-base-uncased/")
roberta_tokenizer = AutoTokenizer.from_pretrained("../../../pre-train-model/roberta/")
# bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class MINDIterator:
    def __init__(self, opt, NPRATIO=-1, t='glove', test=False):
        self.opt = opt
        self.NPRATIO = NPRATIO
        self.word_dict = self.load_dict(Word_dict_PATH)
        self.cat_dict = self.load_dict(Cat_dict_PATH)
        self.cat_emb()
        # ipdb.set_trace()
        self.sub_dict = self.load_dict(Sub_dict_PATH)
        self.uid2index = self.load_dict(User_dict_PATH)
        self.tokenizer = t
        self.test = test

    def cat_emb(self):
        cat_num = len(self.cat_dict)
        embedding = np.random.normal(size=(cat_num+1, 300))
        # ipdb.set_trace()
        emb = np.load(Emb_dict_PATH)
        for x in self.cat_dict.keys():
            index = int(self.cat_dict[x])
            if x in self.word_dict.keys():
                word_index = int(self.word_dict[x])
                embedding[index] = emb[word_index]
            
        np.save(f"{SAVE_PATH}/train/embedding_cat.npy", embedding)

    def newsample(self, news, ratio):
        if ratio > len(news):
            # return random.sample(news * (ratio // len(news) + 1), ratio)
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
        news_ab = ['']
        category = [0]
        sub = [0]
        with open(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split("\t")
                if nid in nid2index:
                    continue
                nid2index[nid] = len(nid2index) + 1
                title = self.wordtoken(title)
                ab = self.wordtoken(ab)
                news_title.append(title)
                news_ab.append(ab)
                category.append(self.cat_dict[vert])
                
                try:
                    sub.append(self.sub_dict[subvert])
                except:
                    ipdb.set_trace()

        def glove_tokenize(news_item, avg_size):
            news_title_index = np.zeros((len(news_item), avg_size),
                                        dtype=np.uint16)
            for news_index in range(len(news_item)):
                title = news_item[news_index]
                for word_index in range(min(avg_size, len(title))):
                    if title[word_index].lower() in word_dict:
                        news_title_index[news_index, word_index] = word_dict[
                            title[word_index].lower()]
            return news_title_index

        def bert_tokenize(news_title, title_size):
            news = [' '.join(t) for t in news_title]
            input_ids = bert_tokenizer(news, max_length=title_size, padding=True, truncation=True)['input_ids']
            return np.array(input_ids, dtype=np.uint16)
        def roberta_tokenize(content_length, list1):
            news = [' '.join(t) for t in list1]
            input_ids = roberta_tokenizer(news, max_length=content_length, padding=True, truncation=True)['input_ids']
            return np.array(input_ids, dtype=np.uint16)

        self.nid2index = nid2index
        if self.tokenizer == 'glove':
            self.news_title_index = glove_tokenize(news_title, title_size)
            self.news_ab_index = glove_tokenize(news_ab, abs_size)
        elif self.tokenizer == 'bert':
            self.news_title_index = bert_tokenize(news_title, title_size)
            self.news_ab_index = bert_tokenize(news_ab, abs_size)
        elif self.tokenizer == 'roberta':
            self.news_title_index = roberta_tokenize(title_size, news_title)
            self.news_ab_index = roberta_tokenize(abs_size, news_ab)
        self.category = category
        self.sub = sub
        return nid2index, self.news_title_index

    def init_behaviors_test(self, behaviors_file, his_size):
        print('\tinit test_behaviors...')
        histories = []
        imprs = []
        impr_indexes = []
        uindexes = []
        his_cats = []
        his_subs = []

        with open(behaviors_file, 'r') as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip('\n').split('\t')[-4:]
                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (his_size - len(history)) + history[-his_size:]
                his_cat = []
                his_sub = []
                for his in history:
                    his_cat.append(self.category[his])
                    his_sub.append(self.sub[his])
                impr_news = [
                    self.nid2index[i] for i in impr.split()
                ]

                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                histories.append(history)
                his_cats.append(his_cat)
                his_subs.append(his_sub)
                imprs.append(impr_news)
                uindexes.append(uindex)
                impr_indexes.append(impr_index)
                impr_index += 1
        self.histories = histories
        self.his_cats = his_cats
        self.his_subs = his_subs
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
        stages = []
        with open(behaviors_file, 'r') as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip('\n').split('\t')[-4:]
                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (his_size - len(history)) + history[-his_size:]
                his_cat = []
                his_sub = []
                for his in history:
                    his_cat.append(self.category[his])
                    his_sub.append(self.sub[his])
                impr_news = [
                    self.nid2index[i.split('-')[0]] for i in impr.split()
                ]

                label = [int(i.split('-')[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                time = int(time.split('/')[1])

                histories.append(history)
                his_cats.append(his_cat)
                his_subs.append(his_sub)
                imprs.append(impr_news)
                labels.append(label)
                uindexes.append(uindex)
                impr_indexes.append(impr_index)
                stages.append(time)
                impr_index += 1
        # ipdb.set_trace()
        max_stages = min(stages)
        stages = [x - max_stages for x in stages]
        self.histories = histories
        self.his_cats = his_cats
        self.his_subs = his_subs
        self.imprs = imprs
        self.labels = labels
        self.uindexes = uindexes
        self.impr_indexes = impr_indexes
        self.stages = stages
        return histories, imprs, labels, uindexes, impr_indexes, stages

    def parse_one_line_test(self, line):
        impr = self.imprs[line]
        click_title_index = self.histories[line]
        click_cat_index = self.his_cats[line]
        click_sub_index = self.his_subs[line]

        candidate_title_index = []
        candidate_cat_index = []
        candidate_sub_index = []
        for news in impr:
            candidate_title_index.append(news)
            candidate_cat_index.append(self.category[news])
            candidate_sub_index.append(self.sub[news])
        yield (click_title_index, click_cat_index, click_sub_index,
               candidate_title_index, candidate_cat_index, candidate_sub_index)

    def parse_one_line(self, line):
        if self.NPRATIO > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]
            stage = self.stages[line]
            uid = self.uindexes[line]
            # ipdb.set_trace()
            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                label = [1] + [0] * self.NPRATIO
                n = self.newsample(negs, self.NPRATIO)
                candidate_title_index = [p] + n
                candidate_cat_index = []
                candidate_sub_index = []
                for i in [p]+n:
                    candidate_cat_index.append(self.category[i])
                    candidate_sub_index.append(self.sub[i])
                click_title_index = self.histories[line]
                click_cat_index = self.his_cats[line]
                click_sub_index = self.his_subs[line]

                yield (click_title_index, click_cat_index, click_sub_index,
                       candidate_title_index, candidate_cat_index, candidate_sub_index, stage, label, uid)
        else:
            news_labels = self.labels[line]
            impr = self.imprs[line]
            uid = self.uindexes[line]
            # ipdb.set_trace()
            click_title_index = self.histories[line]
            click_cat_index = self.his_cats[line]
            click_sub_index = self.his_subs[line]

            candidate_title_index = []
            candidate_cat_index = []
            candidate_sub_index = []
            for news in impr:
                candidate_title_index.append(news)
                candidate_cat_index.append(self.category[news])
                candidate_sub_index.append(self.sub[news])
            yield (click_title_index, click_cat_index, click_sub_index,
                   candidate_title_index, candidate_cat_index, candidate_sub_index, news_labels, uid)
    
    def init_popularity(self):
        # ipdb.set_trace()
        news_num = len(self.sub)
        stages_num = int(max(self.stages)) + 1 + 1
        cat_num = len(self.cat_dict.keys())
        
        popularity = np.zeros([news_num, stages_num])
        cat_popularity = np.zeros([cat_num, stages_num])
        assert len(self.stages) == len(self.imprs) == len(self.labels)
        for x in range(len(self.stages)):
            stage = self.stages[x] + 1
            impr = self.imprs[x]
            label = self.labels[x]
            for y in range(len(label)):
                if label[y] == 1:
                    # ipdb.set_trace()
                    uid = self.category[impr[y]]
                    popularity[impr[y]][stage] += 1
                    cat_popularity[uid][stage] += 1
        # ipdb.set_trace()
        for x in self.histories:
            for y in x:
                if y == 0:
                    continue
                else:
                    popularity[y][0] += 1
        self.popularity = popularity
        self.cat_popularity = cat_popularity
    
    def init_exc_click(self):
        news_num = len(self.sub)
        pop_matrix = self.popularity[:,0]
        mid = np.sum(pop_matrix) / (2 * news_num)
        excClickDict = np.zeros([news_num, 2]).astype('int')
        dict1 = {}
        for x in range(news_num):
            if(pop_matrix[x] <= mid or x == 0):
                excClickDict[x] = [x, x]
                news_sub = self.sub[x]
                if news_sub not in dict1.keys():
                    dict1[news_sub] = []
                dict1[news_sub].append(x)

        for x in range(news_num):
            if(pop_matrix[x] > mid):
                news_sub = self.sub[x]
                if(news_sub not in dict1.keys()):
                    excClickDict[x] = [x, x]
                else:
                    excClickDict[x][0] = random.sample(dict1[news_sub], 1)[0]
                    excClickDict[x][1] = x
        # ipdb.set_trace()
        self.excClickDict = excClickDict



        # ipdb.set_trace()

    def load_data_from_file(self, news_file, behavior_file, mode='Train'):
        if self.test:
            candidate_news_indexes = []
            candidate_cat_indexes = []
            candidate_sub_indexes = []

            click_news_indexes = []
            click_cat_indexes = []
            click_sub_indexes = []
            self.init_news(news_file, self.opt.title_size, self.opt.abs_size, self.word_dict)
            self.init_behaviors_test(behavior_file, self.opt.his_size)
            ipdb.set_trace()
            for i in range(len(self.imprs)):
                for click_news_index, click_cat_index, click_sub_index, candidate_news_index, candidate_cat_index, candidate_sub_index, in self.parse_one_line_test(i):
                    click_news_indexes.append(click_news_index)
                    click_cat_indexes.append(click_cat_index)
                    click_sub_indexes.append(click_sub_index)

                    candidate_news_indexes.append(candidate_news_index)
                    candidate_cat_indexes.append(candidate_cat_index)
                    candidate_sub_indexes.append(candidate_sub_index)
            path = f'{SAVE_PATH}/test'
            print('\t start saving...')
            np.save(f"{path}/click_news_indexes.npy", np.array(click_news_indexes, dtype=np.int32))
            np.save(f"{path}/click_cat_indexes.npy", np.array(click_cat_indexes, dtype=np.int16))
            np.save(f"{path}/click_sub_indexes.npy", np.array(click_sub_indexes, dtype=np.int16))

            np.save(f"{path}/news_title_index_{token_type}.npy", self.news_title_index)
            np.save(f"{path}/news_sapo_index_{token_type}.npy", self.news_ab_index)
            np.save(f"{path}/news_cat_index.npy", self.category)
            np.save(f"{path}/news_sub_index.npy", self.sub)

            with open(f"{path}/candidate_news_indexes.pkl", 'wb') as f:
                pickle.dump(candidate_news_indexes, f)
        else:
            if self.NPRATIO > 0:
                label_list = []
                click_news_indexes = []
                click_cat_indexes = []
                click_sub_indexes = []
                candidate_news_indexes = []
                candidate_cat_indexes = []
                candidate_sub_indexes = []
                click_stages = []
                uindexes_list = []
                self.init_news(news_file, self.opt.title_size, self.opt.abs_size, self.word_dict)
                self.init_behaviors(behavior_file, self.opt.his_size)
                self.init_popularity()
                self.init_exc_click()

                for i in range(len(self.imprs)):
                    for click_news_index, click_cat_index, click_sub_index, candidate_news_index, candidate_cat_index, candidate_sub_index, stage, label, uid in self.parse_one_line(i):
                        click_news_indexes.append(click_news_index)
                        click_cat_indexes.append(click_cat_index)
                        click_sub_indexes.append(click_sub_index)
                        click_stages.append(stage)
                        uindexes_list.append(uid)
                        candidate_news_indexes.append(candidate_news_index)
                        candidate_cat_indexes.append(candidate_cat_index)
                        candidate_sub_indexes.append(candidate_sub_index)

                        label_list.append(label)
                path = f'{SAVE_PATH}/train'

                print('\t start saving...')
                np.save(f"{path}/click_news_indexes.npy", np.array(click_news_indexes, dtype=np.int32))
                np.save(f"{path}/click_cat_indexes.npy", np.array(click_cat_indexes, dtype=np.int16))
                np.save(f"{path}/click_sub_indexes.npy", np.array(click_sub_indexes, dtype=np.int16))

                np.save(f"{path}/news_title_index_{token_type}.npy", self.news_title_index)
                np.save(f"{path}/news_sapo_index_{token_type}.npy", self.news_ab_index)
                np.save(f"{path}/news_cat_index.npy", self.category)
                np.save(f"{path}/news_sub_index.npy", self.sub)

                np.save(f"{path}/candidate_news_indexes.npy", np.array(candidate_news_indexes, dtype=np.int32))
                np.save(f"{path}/candidate_cat_indexes.npy", np.array(candidate_cat_indexes, dtype=np.int16))
                np.save(f"{path}/candidate_sub_indexes.npy", np.array(candidate_sub_indexes, dtype=np.int16))

                np.save(f"{path}/news_popularity.npy", self.popularity)
                np.save(f"{path}/exc_click_dict.npy", self.excClickDict)
                np.save(f"{path}/click_stages.npy", np.array(click_stages, dtype=np.int16))
                np.save(f"{path}/cat_popularity.npy", np.array(self.cat_popularity, dtype=np.int32))

                np.save(f"{path}/label_list.npy", label_list)
                np.save(f"{path}/uindexes_list.npy", uindexes_list)
                # np.save(f"{path}/click_stage.npy", self.stages)
            else:
                label_list = []
                candidate_news_indexes = []
                click_news_indexes = []
                candidate_cat_indexes = []
                candidate_sub_indexes = []
                click_cat_indexes = []
                click_sub_indexes = []
                uindexes_list = []

                self.init_news(news_file, self.opt.title_size, self.opt.abs_size, self.word_dict)
                self.init_behaviors(behavior_file, self.opt.his_size)
                # ipdb.set_trace()
                news_num = len(self.sub)
                pop_matrix = np.zeros([news_num])
                
                for i in range(len(self.imprs)):
                    for click_news_index,  click_cat_index, click_sub_index, candidate_news_index, candidate_cat_index, candidate_sub_index, label, uid in self.parse_one_line(i):
                        click_news_indexes.append(click_news_index)
                        click_cat_indexes.append(click_cat_index)
                        click_sub_indexes.append(click_sub_index)
                        uindexes_list.append(uid)
                        candidate_news_indexes.append(candidate_news_index)
                        candidate_cat_indexes.append(candidate_cat_index)
                        candidate_sub_indexes.append(candidate_sub_index)

                        label_list.append(label)
                # ipdb.set_trace()
                path = f'{SAVE_PATH}/dev'
                for (news, label) in zip(candidate_news_indexes, label_list):
                    for (uid, is_click) in zip(news, label):
                        if is_click == 1:
                            pop_matrix[uid] += 1
                print('\t start saving...')
                np.save(f"{path}/click_news_indexes.npy", np.array(click_news_indexes, dtype=np.int32))
                np.save(f"{path}/click_cat_indexes.npy", np.array(click_cat_indexes, dtype=np.int16))
                np.save(f"{path}/click_sub_indexes.npy", np.array(click_sub_indexes, dtype=np.int16))

                np.save(f"{path}/news_title_index_{token_type}.npy", self.news_title_index)
                np.save(f"{path}/news_sapo_index_{token_type}.npy", self.news_ab_index)
                np.save(f"{path}/news_cat_index.npy", self.category)
                np.save(f"{path}/news_sub_index.npy", self.sub)
                np.save(f"{path}/news_popularity.npy", pop_matrix)
                np.save(f"{path}/uindexes_list.npy", uindexes_list)


                with open(f"{path}/candidate_news_indexes.pkl", 'wb') as f:
                    pickle.dump(candidate_news_indexes, f)
                with open(f"{path}/candidate_cat_indexes.pkl", 'wb') as f:
                    pickle.dump(candidate_cat_indexes, f)
                with open(f"{path}/candidate_sub_indexes.pkl", 'wb') as f:
                    pickle.dump(candidate_sub_indexes, f)

                with open(f"{path}/label_list.pkl", 'wb') as f:
                    pickle.dump(label_list, f)


if __name__ == "__main__":
    assert len(sys.argv) == 3, 'please specify the data type and glove/bert: \n\t python3 data_pro.py large glove'
    data_type = sys.argv[1]
    token_type = sys.argv[2]

    print(f"dataset: {data_type}; use {token_type} as tokenizer")

    assert data_type in ['large', 'small', 'llarge']
    assert token_type in ['glove', 'bert', 'roberta']

    if data_type == 'large':
        DATA_PATH = './raw/large'
        SAVE_PATH = '../data/large'
        User_dict_PATH = './raw/uid2index_large.pkl'
        Cat_dict_PATH = './raw/category2index_large.pkl'
        Sub_dict_PATH = './raw/subcategory2index_large.pkl'
        Word_dict_PATH = './raw/word_dict_large.pkl'
        Emb_dict_PATH = './raw/embedding_large.npy'
    elif data_type == 'small':
        DATA_PATH = './raw/small'
        SAVE_PATH = '../data/small'
        User_dict_PATH = './raw/uid2index_small.pkl'
        Cat_dict_PATH = './raw/category2index_small.pkl'
        Sub_dict_PATH = './raw/subcategory2index_small.pkl'
        Word_dict_PATH = './raw/word_dict_small.pkl'
        Emb_dict_PATH = './raw/embedding_small.npy'
    else:
        DATA_PATH = './raw/llarge'
        SAVE_PATH = '../data/llarge'
        User_dict_PATH = './raw/uid2index_llarge.pkl'
        Cat_dict_PATH = './raw/category2index_llarge.pkl'
        Sub_dict_PATH = './raw/subcategory2index_llarge.pkl'

    # train_iterator = MINDIterator(opt, NPRATIO=opt.ratio_K, t=token_type)
    # print("process train data...")
    # train_users = train_iterator.load_data_from_file(
    #     f"{DATA_PATH}/train/news.tsv", f"{DATA_PATH}/train/behaviors.tsv", 'Train')

    dev_iterator = MINDIterator(opt, t=token_type)
    print("process dev data...")
    dev_iterator.load_data_from_file(
        f"{DATA_PATH}/dev/news.tsv", f"{DATA_PATH}/dev/behaviors.tsv", 'Dev')

    if data_type == 'large':
        test_iterator = MINDIterator(opt, t=token_type, test=True)
        print("process test data...")
        test_iterator.load_data_from_file(
            f"{DATA_PATH}/test/news.tsv", f"{DATA_PATH}/test/behaviors.tsv", 'Test')

######### small ##########
# train: 2019-11-09:2019-11-14 (156965)
# test:  2019-11-15            (73152)

######### large ##########
# train: 2019-11-09:2019-11-14 (2232748)
# dev:   2019-11-15            (376471)
# test:  2019-11-16:2019-11-22 (2370727)