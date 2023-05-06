import pandas as pd
import numpy as np
import json
import random
import os
import sys
import pickle
from easydict import EasyDict as edict
import time
from datetime import datetime
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import config as train_config
from dataset import get_dataloader
from util import save_checkpoint, one_hot, iter_product, clip_gradient, load_model
from sklearn.metrics import accuracy_score, f1_score
import loss as loss
from model import contextual_encoder
import math

from transformers import AdamW, get_linear_schedule_with_warmup


def train(epoch, train_loader, model_main, loss_function, optimizer, lr_scheduler, log):

    model_main.cuda()
    model_main.train()

    total_true, total_pred, acc_curve = [], [], []
    train_loss = 0
    total_epoch_acc = 0
    steps = 0
    start_train_time = time.time()

    if log.param.loss_type == "scl":
        train_batch_size = log.param.batch_size * 2
    else:
        train_batch_size = log.param.batch_size
    for idx, batch in enumerate(train_loader):
        if "bpw" in log.param.dataset:
            text_name = "sentence"
            label_name = "label"

        text = batch[text_name]
        attn = batch[text_name+"_attn_mask"]
        label = batch[label_name]
        label = torch.tensor(label)
        label = torch.autograd.Variable(label).long()

        if (label.size()[0] is not train_batch_size):# Last batch may have length different than log.param.batch_size
            continue

        if torch.cuda.is_available():
            text = text.cuda()
            attn = attn.cuda()
            label = label.cuda()

        pred, supcon_feature = model_main(text, attn)

        if log.param.loss_type == "scl":
            loss = (loss_function["lambda_loss"]*loss_function["label"](pred, label)) + ((1-loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature, label))
        else:
            loss = loss_function["label"](pred, label)

        train_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)

        optimizer.step()
        model_main.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print(f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Time taken: {((time.time() - start_train_time) / 60): .2f} min')
            start_train_time = time.time()

        true_list = label.data.detach().cpu().tolist()
        total_true.extend(true_list)
        num_corrects = (torch.max(pred, 1)[1].view(label.size()).data == label.data).float().sum()
        pred_list = torch.max(pred, 1)[1].view(label.size()).data.detach().cpu().tolist()
        total_pred.extend(pred_list)

        acc = 100.0 * (num_corrects/train_batch_size)
        acc_curve.append(acc.item())
        total_epoch_acc += acc.item()

    return train_loss/len(train_loader), total_epoch_acc/len(train_loader), acc_curve


def test(epoch, test_loader, model_main, loss_function, log):
    model_main.eval()
    total_epoch_acc = 0
    total_pred, total_true, total_pred_prob = [], [], []
    save_pred = {"true": [], "pred": [], "pred_prob": [], "feature": []}
    acc_curve = []
    total_feature = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if "bpw" in log.param.dataset:
                text_name = "sentence"
                label_name = "label"

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]
            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            pred, supcon_feature = model_main(text, attn)
            num_corrects = (torch.max(pred, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list = torch.max(pred, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            acc = 100.0 * num_corrects / 1
            acc_curve.append(acc.item())
            total_epoch_acc += acc.item()

            total_pred.extend(pred_list)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature.data.detach().cpu().tolist())
            total_pred_prob.extend(pred.data.detach().cpu().tolist())

    f1_score_m = f1_score(total_true, total_pred, average="macro")
    f1_score_w = f1_score(total_true, total_pred, average="weighted")

    f1_score_all = {"macro": f1_score_m, "weighted": f1_score_w}

    save_pred["true"] = total_true
    save_pred["pred"] = total_pred

    save_pred["feature"] = total_feature
    save_pred["pred_prob"] = total_pred_prob

    return total_epoch_acc/len(test_loader), f1_score_all, save_pred, acc_curve


def stega_train(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    train_data, valid_data, test_data = get_dataloader(log.param.batch_size, log.param.corpus, log.param.stego_method, log.param.dataset, w_aug=log.param.is_waug)

    if log.param.loss_type == "scl":
        losses = {"contrastive": loss.SupConLoss(temperature=log.param.temperature), "label": nn.CrossEntropyLoss(), "lambda_loss": log.param.lambda_loss}
    else:
        losses = {"label": nn.CrossEntropyLoss(), "lambda_loss": log.param.lambda_loss, "contrastive": loss.SupConLoss(temperature=log.param.temperature)}

    run_start = datetime.now()
    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model_main = contextual_encoder(log.param.batch_size, log.param.hidden_size, log.param.label_size, log.param.model_type)

    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data)*log.param.nepoch)
    print("num_training_steps: ", num_training_steps)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)], 'weight_decay': log.param.decay},
                                    {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    save_home = "./save/final/"+log.param.corpus+"/"+log.param.stego_method+"/"+log.param.dataset+"/"+log.param.loss_type+"/"+model_run_time+"/"

    total_train_acc_curve, total_val_acc_curve = [], []

    for epoch in range(1, log.param.nepoch + 1):

        train_loss, train_acc, train_acc_curve = train(epoch, train_data, model_main, losses, optimizer, lr_scheduler, log)
        val_acc, val_f1, val_save_pred, val_acc_curve = test(epoch, valid_data, model_main, losses, log)
        test_acc, test_f1, test_save_pred, test_acc_curve = test(epoch, test_data, model_main, losses, log)

        total_train_acc_curve.extend(train_acc_curve)
        total_val_acc_curve.extend(val_acc_curve)

        print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        os.makedirs(save_home,exist_ok=True)
        with open(save_home+"/acc_curve.json", 'w') as fp:
            json.dump({"train_acc_curve": total_train_acc_curve, "val_acc_curve": total_val_acc_curve}, fp, indent=4)
        fp.close()

        if epoch == 1:
             best_criterion = 0
        is_best = val_acc > best_criterion
        best_criterion = max(val_acc, best_criterion)

        print(f'Valid Accuracy: {val_acc:.2f}  Valid F1: {val_f1["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc:.2f}  Test F1: {test_f1["macro"]:.2f}')

        if is_best:
            print("======> Best epoch <======")
            log.train_loss = train_loss
            log.stop_epoch = epoch
            log.valid_f1_score = val_f1
            log.test_f1_score = test_f1
            log.valid_accuracy = val_acc
            log.test_accuracy = test_acc
            log.train_accuracy = train_acc

            ## save the model
            # torch.save(model_main.state_dict(), save_home+'best.pt')
            run_end = datetime.now()
            best_time = str((run_end - run_start).seconds / 60) + ' minutes'
            log.best_time = best_time

            with open(save_home+"/log.json", 'w') as fp:
                json.dump(dict(log), fp, indent=4)
            fp.close()

            with open(save_home+"/feature.json", 'w') as fp:
                json.dump(test_save_pred, fp, indent=4)
            fp.close()


if __name__ == '__main__':

    tuning_param = train_config.tuning_param
    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num, val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        if "allbpw" in log.param.dataset:
            log.param.label_size = 6
        elif "bpw" in log.param.dataset:
            log.param.label_size = 2

        run_start = datetime.now()
        stega_train(log)
        run_end = datetime.now()
        run_time = str((run_end - run_start).seconds / 60) + ' minutes'
        print("corpus: ", log.param.corpus, "stego_method: ", log.param.stego_method, "dataset: ", log.param.dataset, "model_type: ", log.param.model_type, "run_time: ", run_time)


