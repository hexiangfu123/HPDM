#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

res_path = '08-03_20:16:53_group_preds.npy'
file_path = './prediction.txt'


def res_pro(res_name, file_name):
    pred = np.load(res_path, allow_pickle=True)
    with open(file_path, 'w') as f:
        imp_index = 1
        for pre in pred:
            f.write(str(imp_index))
            f.write(' [')
            sortdict = {}
            # __import__('ipdb').set_trace()
            for i, j in enumerate(pre):
                sortdict[i] = j
            ansdict = sorted(sortdict.items(), key=lambda item: item[1], reverse=True)

            anslist = ansdict.copy()
            for index in range(len(ansdict)):
                anslist[ansdict[index][0]] = index + 1
            for index in range(len(anslist)-1):
                f.write(str(anslist[index]))
                f.write(',')
            f.write(str(anslist[-1]))
            f.write(']\n')
            imp_index += 1


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'python3 result_process.py ***group_preds.npy'
    res_path = sys.argv[1]
    res_pro(res_path, file_path)
