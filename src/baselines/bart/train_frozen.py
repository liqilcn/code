import re, itertools,os, sys, time, requests,torch,random,logging,argparse,json,datetime,heapq
from tqdm import tqdm
import pandas as pd
import numpy as np
from os.path import join, exists
from torch.utils.data import Dataset,DataLoader,random_split 
from torch.nn.functional import softmax
import torch.nn as nn
import pickle as pkl


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
        # self.dense = nn.Linear(input_dim, inner_dim)
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
        
        # hidden_states = self.dense(hidden_states)
        # hidden_states = torch.relu(hidden_states) 

        hidden_states = self.out_proj(hidden_states)

        return hidden_states


def setup_seed(seed=1996):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_accuracy(predicted_top2, top2_gd):  
    # hit@2指标
    sentence_num = sum([len(rgt) for rgt in top2_gd])
    correct_num = 0
    for i in range(len(predicted_top2)):
        correct_num += len(set(predicted_top2[i])&set(top2_gd[i]))
    return round(float(correct_num)/float(sentence_num), 8)*100

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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device
        labels = inputs['labels']
        outputs = model(**inputs)
        # todo: weight: 正样本权重越高， recall也高， 
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["learning_rate"] = self._get_learning_rate()
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        labels = inputs[self.label_names[0]].cuda()
        inputs = {k:v.cuda() for k,v in inputs.items()}
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            logits = outputs
        return (loss, logits, labels)

def compute_metrics(p):
    report = {}
    preds,labels = p
    pred_labels  = np.argmax(preds, axis=-1)
    y_true = np.squeeze(labels)
    predict = scipy.special.softmax(preds.data, 1)
    y_score = predict[:,1]
    auc = compute_auc(y_true, y_score)
    fpr95 = compute_fpr95(y_true, y_score)
    relation_num = sum([1 for rgt in y_true])
    correct_num = 0
    for i in range(len(pred_labels)):
        correct_num += 1 if pred_labels[i]== y_true[i] else 0
    acc = round(float(correct_num)/float(relation_num)*100,4)
    report['acc'] = acc
    report['auc'] = auc
    report['fpr95'] = fpr95
    loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
    loss = loss_fct(torch.tensor(preds.reshape(-1, 2)).to(device), torch.tensor(y_true.reshape(-1)).to(device))    
    report['loss'] = loss.item()   
    print('\n')
    for k ,v in report.items(): print(f'*-->{k:20}: {v}')
    print('\n')
    report = {k:v for k ,v in report.items() if not isinstance(v, dict)}
    return report

if __name__ == "__main__":
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='large', type=str)
    flags.add_argument('-seed',                default=2023, type=int)

    # flags.add_argument('-dataset_path',        default='./4to1_dataset/', type=str)
    flags.add_argument('-dataset_type',        default='delve_1k', type=str)
    flags.add_argument('-tokenizer_path',      default='./tokenizer_config/', type=str)
    flags.add_argument('-init_model_path',     default='./init_model/', type=str)
    flags.add_argument('-decoder_layer',       default=11,    type=int)#与encoder信息融合的decoder层
    
    flags.add_argument('-max_src_len',         default=1024,    type=int)
    flags.add_argument('-max_tgt_len',         default=100,     type=int)
    flags.add_argument('-max_doc_len',         default=250,     type=int)

    flags.add_argument('-batch_size',          default=64,    type=int)
    flags.add_argument('-gradient_accumulation_steps', default=1, type=int)
    flags.add_argument('-lr',                  default=1e-3,  type=float)  # BART 3e-5 达到最佳
    flags.add_argument('-warmup_steps',        default=200,   type=int)
    flags.add_argument('-weight_decay',        default=1e-4,  type=float)
    flags.add_argument('-epochs',              default=40,    type=int)
    flags.add_argument('-num_workers',         default=4,    type=int)
    flags.add_argument('-save_steps',          default=100,     type=int)
    flags.add_argument('-evaluation_strategy', default='epoch',     type=str)
    flags.add_argument('-eval_steps',          default=100,     type=int)
    flags.add_argument('-log_steps',           default=10,     type=int)
    flags.add_argument('-mode',                default='train', type=str)
    flags.add_argument('-output_dir',          default='./ckpts/', type=str)
    
    
    args, unknown   = flags.parse_known_args()
    print(args.model_size)
    print(args.dataset_type)
    setup_seed(args.seed)
    model_size = args.model_size # 选择base或者large
    dataset_type = args.dataset_type
    tag = f'{args.dataset_type}_{str(args.lr)}_{args.epochs}_{args.batch_size}_' + \
        f'{args.seed}'
        # f'{args.gradient_accumulation_steps}_{args.num_workers}'
    args.output_dir = f'{args.output_dir}freeze_new/{model_size}/'+tag
    args.tokenizer_path = args.tokenizer_path + model_size
    # args.init_model_path = f'{args.init_model_path}{model_size}/' + dataset_type
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    
    model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
    model_config.max_length = args.max_tgt_len
    model_config.decoder_layer = args.decoder_layer
    model_config.num_labels = 2
    d_model = model_config.d_model
    

    
    # load保存好的embedding
    embed_path = f'embedding/{model_size}/' + dataset_type
    train_emb_dataset = pkl.load(open(f'{embed_path}/train.pkl', 'rb'))
    valid_emb_dataset = pkl.load(open(f'{embed_path}/valid.pkl', 'rb'))


    # 两层全连接层 训练
    # print(d_model*2)
    lnn_model = SupervisedLNN(d_model*2,d_model*2,2,0)
    lnn_model.cuda()
    device = next(lnn_model.parameters()).device

    trainer_args = TrainingArguments(
            # no_cuda=False,
            output_dir=args.output_dir,
            logging_dir='%s/log_%s' % (args.output_dir, tag),
            overwrite_output_dir=True,
            # train config
            # label_names=TARGE_NAMES, # if set variable 'label_names', compute_metric will not run;
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # max_steps=200,
            # eval
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=int(args.batch_size*8),
            # optimizer
            learning_rate=args.lr,
            lr_scheduler_type='constant_with_warmup', # "linear, cosine, cosine_with_restarts, polynomial, constant_with_warmup"
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            adafactor=False,
            # label_smoothing_factor=0.1,
            # log
            log_on_each_node=True,
            logging_strategy='steps',
            logging_steps=args.log_steps,
            logging_first_step=True,
            report_to='all',
            # model resume and save
            # resume_from_checkpoint="./ckpts/pretrained",
            save_strategy='epoch',
            # save_steps=args.save_steps,
            save_total_limit=50,
            # load_best_model_at_end=True,
            # metric_for_best_model='eval_f1',
            ignore_data_skip=True,
            # others
            dataloader_num_workers=args.num_workers,
            dataloader_drop_last=False,
            dataloader_pin_memory=True,
            seed=args.seed)
    trainer = CustomTrainer(
                    model          =lnn_model,
                    args           =trainer_args,
                    train_dataset  =train_emb_dataset,
                    eval_dataset   =valid_emb_dataset,
                    compute_metrics=compute_metrics)

    trainer.train() 
    
