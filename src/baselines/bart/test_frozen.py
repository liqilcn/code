#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, itertools,os, sys, time, requests,torch,random,logging,argparse,json,datetime,heapq
from tqdm import tqdm
import pandas as pd
import numpy as np
from os.path import join, exists
from torch.utils.data import Dataset,DataLoader,random_split 
from torch.nn.functional import softmax
import torch.nn as nn
import pickle as pkl
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers import (BartTokenizer, 
    BartModel,
    BartPretrainedModel,
    BartConfig,
    TrainingArguments,
    Trainer,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup)
import scipy
from sklearn.metrics import precision_recall_fscore_support,classification_report,precision_recall_curve
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import interpolate
from typing import Any, Dict, List, Optional, Set, Tuple, Union
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"


class SupervisedLNN(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, inner_dim*4)
        self.dense2 = nn.Linear(inner_dim*4, inner_dim*2)
        self.dense3 = nn.Linear(inner_dim*2, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor,labels) -> torch.Tensor:
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = torch.relu(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense3(hidden_states)
        hidden_states = torch.relu(hidden_states) 

        hidden_states = self.out_proj(hidden_states)

        return hidden_states



def compute_aupr(y_true, y_score):
    aupr = average_precision_score(y_true, y_score)
    return round(aupr,8)

def compute_auc(y_true, y_score):
    #计算auc
    #score是label对应的分数
    auc = roc_auc_score(y_true, y_score)
    return round(auc,8)

def compute_fpr95(y_true, y_score):
    #计算fpr95
    fpr,tpr,thres = roc_curve(y_true, y_score, pos_label=1)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    return round(fpr95,8)


class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    def __getitem__(self, index):
        return {'hidden_states':self.data[index][0],'labels':self.data[index][1]}
    def __len__(self): 
        return len(self.data)


class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    def __getitem__(self, index):
        return {'hidden_states':self.data[index][0],'labels':self.data[index][1]}
    def __len__(self): 
        return len(self.data)


# In[6]:


def get_data_log(ckpt_pth):
    file_list = os.listdir(ckpt_pth)
    ids = [int(item.split('-')[-1]) for item in file_list if 'checkpoint' in item]
    index = max(ids)
    
    return f'checkpoint-{index}'


def get_best_step(data_log):
    val_step, val_auc = [], []
    for item in data_log:
        if "eval_loss" in item.keys():
            val_auc.append(item["eval_auc"])
            val_step.append(item['step'])
    
    max_auc = max(val_auc)
    max_index = val_auc.index(max_auc)
    return max_auc, val_step[max_index]


# In[7]:


def save_test_res(test_dir, init_model_path):
    model_config = BartConfig.from_pretrained(init_model_path, local_files_only=True)

    d_model = model_config.d_model
    

    output_detail = {
    'predicted_label':[],
    'groundtruth_label':[],
    'predicted_score':[]
    }
    # model = nn.DataParallel(model)
    ckpt_path = f'{test_dir}/pytorch_model.bin' 
    pretrained_state = torch.load(ckpt_path)
    print('load model')
    
    test_model = SupervisedLNN(d_model*2,d_model*2,2,0)
    test_model.load_state_dict(pretrained_state ,strict=False)
    test_model.eval()
    with torch.no_grad():
        for data in tqdm(test_emb_dataset):
            outputs = test_model(**data)
            predict = torch.softmax(outputs.data, 0).squeeze(0).cpu().numpy().tolist() 
            predicted_score = predict[1]
            predicted_label = int(np.argmax(predict))
            
            output_detail['predicted_score'].append(predicted_score)
            output_detail['predicted_label'].append(predicted_label)
            output_detail['groundtruth_label'].append(int(data['labels'][0]))

        aupr = compute_aupr(output_detail['groundtruth_label'],output_detail['predicted_score'])
        auc = compute_auc(output_detail['groundtruth_label'],output_detail['predicted_score'])
        fpr95 = compute_fpr95(output_detail['groundtruth_label'],output_detail['predicted_score'])
        report = {'fpr95':fpr95, 'auroc':auc, 'aupr':aupr}
    #     print(report)
    # return report
        parameter = test_dir.split('/')[-2]
    return report, parameter



def get_metrics(ckpt_pth):
    
    file_list = os.listdir(ckpt_pth)
    ids = [int(item.split('-')[-1]) for item in file_list if 'checkpoint' in item]
    file_pth = f'checkpoint-{max(ids)}'
    
    data_log = json.load(open(ckpt_pth+'/'+file_pth+'/trainer_state.json', 'rb'))

    max_auc, best_step = get_best_step(data_log["log_history"])
    test_dir = ckpt_pth + '/' + f'checkpoint-{best_step}'

    return max_auc, test_dir



def get_auc_dir(model_data_path,init_model_pth):

    max_auc, selected_dir = get_metrics(model_data_path)
    res, rdir = save_test_res(best_dir,init_model_pth)
    return res, rdir


if __name__ == "__main__":
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='large', type=str)
    flags.add_argument('-dataset_type',        default='delve_1k', type=str)
    flags.add_argument('-ckpts_path',          default='', type=str)  # 单次训练Frozen的FNN所产生的所有ckpts

    embed_path = f'./embedding/{args.model_size}/{args.dataset_type}'
    test_emb_dataset = pkl.load(open(f'{embed_path}/test.pkl', 'rb'))
    init_model_pth = f'./init_model/{args.model_size}'
    res, rdir = get_auc_dir(args.ckpts_path, init_model_pth)
    if os.path.exists('./frozen_test_results/'): os.makedirs('./frozen_test_results/')
    json.dump(res, open(f'./frozen_test_results/{args.model_size}_{rdir}.json', 'w'))
    print(res)
