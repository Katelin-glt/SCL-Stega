import pandas as pd
import numpy as np
import json
import random
import os
import sys
import pickle
import itertools

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from collate_fns import collate_fn_w_aug, collate_fn


class Stega_dataset(Dataset):

    def __init__(self, data, training=True, w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug

    def __getitem__(self, index):
        item = {}

        if self.training and self.w_aug:
            item["sentence"] = self.data["tokenized_sentence"][index]
        else:
            item["sentence"] = torch.LongTensor(self.data["tokenized_sentence"][index])
        item["label"] = self.data["label"][index]

        return item

    def __len__(self):
        return len(self.data["label"])


def get_dataloader(batch_size, corpus, stego_method, dataset, seed=None, w_aug=True, label_list=None):

    if w_aug:
        with open('./preprocessed_data/'+corpus+"/"+stego_method+"/"+dataset+'_waug_preprocessed_bert.pkl', "rb") as f:
            data = pickle.load(f)
            f.close()
    else:
        print("============================== No Aug ===========================================")
        with open('./preprocessed_data/'+corpus+"/"+stego_method+"/"+dataset+'_preprocessed_bert.pkl', "rb") as f:
            data = pickle.load(f)
        f.close()

    train_dataset = Stega_dataset(data["train"], training=True, w_aug=w_aug)
    valid_dataset = Stega_dataset(data["dev"], training=False, w_aug=w_aug)
    test_dataset = Stega_dataset(data["test"], training=False, w_aug=w_aug)

    if w_aug:
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_w_aug, num_workers=0)
    else:
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return train_iter, valid_iter, test_iter

