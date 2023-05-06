import sys

import pandas as pd
import numpy as np
import json
import random
import pickle
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import ElectraModel, BertModel, AutoModel
from util import load_model


class contextual_encoder(nn.Module):

    def __init__(self, batch_size, hidden_size, label_size, encoder_type="electra"):
        super(contextual_encoder, self).__init__()

        if encoder_type == "electra":
            options_name = "google/electra-base-discriminator"
            self.encoder_supcon = ElectraModel.from_pretrained(options_name, num_labels=label_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing = True
        elif encoder_type == "bert":
            self.encoder_supcon = BertModel.from_pretrained('bert-base-uncased', num_labels=label_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing = True

        self.pooler_fc = nn.Linear(hidden_size, hidden_size)  # fully-connected layer
        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size,label_size)  # output logits

    def get_emedding(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.pooler_dropout(x)
        x = F.relu(x)
        return x

    def forward(self, text, attn_mask):
        supcon_fea = self.encoder_supcon(text, attn_mask, output_hidden_states=True, output_attentions=True, return_dict=True)
        supcon_fea_cls_logits = self.get_emedding(supcon_fea.hidden_states[-1])
        supcon_fea_cls_logits = self.pooler_dropout(self.label(supcon_fea_cls_logits))

        supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:, 0, :], dim=1)
        return supcon_fea_cls_logits, supcon_fea_cls

