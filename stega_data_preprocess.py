import sys

import pandas as pd
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
import random
from sklearn.model_selection import train_test_split

import torch
from transformers import BertTokenizer,AutoTokenizer


def preprocess_data(corpus, stego_method, dataset,tokenizer_type,w_aug=True):
        print("Extracting data")
        data_home = "./data/Steganalysis/"+corpus+"/"+stego_method+"/"+dataset+"/"

        data_dict = {}
        for datatype in ["train", "dev", "test"]:
            if datatype == "train" and w_aug:
                data = pd.read_csv(data_home+datatype+".csv")
                data.dropna()
                final_sentence, final_label = [], []
                for i,val in enumerate(data["label"]):
                    final_sentence.append(data["sentence"][i])
                    final_label.append(val)

                augmented_sentence = list(data["augment"])

                print("Tokenizing data")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
                tokenized_sentence_original = tokenizer.batch_encode_plus(final_sentence).input_ids
                tokenized_sentence_augmented = tokenizer.batch_encode_plus(augmented_sentence).input_ids

                tokenized_combined_sentence = [list(i) for i in zip(tokenized_sentence_original, tokenized_sentence_augmented)]
                combined_sentence = [list(i) for i in zip(final_sentence, augmented_sentence)]
                combined_label = [list(i) for i in zip(final_label, final_label)]

                processed_data = {}

                # ## changed sentence --> sentence for uniformity
                processed_data["tokenized_sentence"] = tokenized_combined_sentence
                processed_data["label"] = combined_label
                processed_data["sentence"] = combined_sentence

                data_dict[datatype] = processed_data

            else:
                data = pd.read_csv(data_home+datatype+".csv", header=0)
                data.dropna()
                final_sentence,final_label = [],[]
                for i,val in enumerate(data["label"]):

                    final_sentence.append(data["sentence"][i])
                    final_label.append(val)

                print("Tokenizing data")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
                tokenized_sentence_original = tokenizer.batch_encode_plus(final_sentence).input_ids

                processed_data = {}
                processed_data["tokenized_sentence"] = tokenized_sentence_original
                processed_data["label"] = final_label
                processed_data["sentence"] = final_sentence
                data_dict[datatype] = processed_data

            if w_aug:
                with open("./preprocessed_data/"+corpus+"/"+stego_method+"/"+dataset+"_waug_preprocessed_bert.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)
                f.close()
            else:
                with open("./preprocessed_data/"+corpus+"/"+stego_method+"/"+dataset+"_preprocessed_bert.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)
                f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Enter tokenizer type')
    parser.add_argument('-corpus', default="Twitter", type=str, help='Enter dataset=')
    parser.add_argument('-stego_method', default="VLC", type=str, help='Enter dataset=')
    parser.add_argument('-d', default="1bpw",type=str, help='Enter dataset=')
    parser.add_argument('-t', default="bert-base-uncased", type=str, help='Enter tokenizer type')
    parser.add_argument('--aug', default=True, action='store_true')

    args = parser.parse_args()

    preprocess_data(args.corpus, args.stego_method, args.d, args.t, w_aug=args.aug)
