# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pdb

test = './raw_1/large/test/entity_embedding.vec'
dev = './raw_1/large/dev/entity_embedding.vec'
train = './raw_1/large/train/entity_embedding.vec'
paths = [test, dev, train]
save_path = './raw_1/entityid2index_large.pkl'
emb_path = './raw_1/entity_embedding.npy'

en2id = {}
id2emb = [[0.]*100]
num = 1
for path in paths:
    with open(path, 'r') as f:
        for line in f.readlines():
            line_ = line.strip().split('\t')
            if line_[0] not in en2id.keys():
                en2id[line_[0]] = num
                id2emb.append(line_[1:])
                num += 1
with open(save_path, 'wb') as ff:
    pickle.dump(en2id, ff)

# save entity embedding
id2emb = np.array(id2emb, dtype='float32')
np.save(emb_path, id2emb)
