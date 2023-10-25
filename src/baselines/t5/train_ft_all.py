# %%
# %%
import re, itertools,os, sys, time, requests,torch,random,logging,argparse,json,datetime,heapq
from tqdm import tqdm
import pandas as pd
import numpy as np
from os.path import join, exists
from torch.utils.data import Dataset,DataLoader,random_split 
from torch.nn.functional import softmax
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers import (T5Tokenizer, 
    T5Model,
    T5PreTrainedModel,
    T5Config,
    TrainingArguments,
    Trainer,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import scipy
from sklearn.metrics import precision_recall_fscore_support,classification_report,precision_recall_curve
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy import interpolate
from typing import Any, Dict, List, Optional, Set, Tuple, Union
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
os.environ["WANDB_DISABLED"] = "true"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# %%
# %%
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = torch.relu(hidden_states)
        # # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense3(hidden_states)
        hidden_states = torch.relu(hidden_states) 
    

        hidden_states = self.out_proj(hidden_states)

        return hidden_states



# %%
class T5ForSequenceClassificationNew(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = T5Model(config)
        self.classification_head = SupervisedLNN(
            config.d_model*2,
            config.d_model*2,
            2,
            0,
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.decoder_layer = config.decoder_layer

        self.model._init_weights(self.classification_head.dense1)
        self.model._init_weights(self.classification_head.dense2)
        self.model._init_weights(self.classification_head.dense3)
        # self.model._init_weights(self.classification_head.dense4)
        self.model._init_weights(self.classification_head.out_proj)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.model.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.model.encoder.block))
        self.model.encoder.parallelize(self.device_map)
        self.model_parallel = True


    def deparallelize(self):
        self.model.encoder.deparallelize()
        self.model.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,#output_hidden_states,
            return_dict=return_dict,
        )

        decoder_all_hidden_states_middle = outputs.decoder_hidden_states_middle
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        decoder_hidden_states_middle = decoder_all_hidden_states_middle[self.decoder_layer]#取第10层的decoder的hidden state出来
        eos_mask_encoder = input_ids.eq(self.config.eos_token_id).to(encoder_last_hidden_state.device)
        eos_mask_decoder = decoder_input_ids.eq(self.config.eos_token_id).to(decoder_hidden_states_middle.device)

        if len(torch.unique_consecutive(eos_mask_encoder.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        if len(torch.unique_consecutive(eos_mask_decoder.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")


        encoder_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_last_hidden_state.size()).float()
        decoder_mask_expanded = decoder_attention_mask.unsqueeze(-1).expand(decoder_hidden_states_middle.size()).float()

        encoder_sum_embeddings = torch.sum(encoder_last_hidden_state * encoder_mask_expanded, 1)
        decoder_sum_embeddings = torch.sum(decoder_hidden_states_middle * decoder_mask_expanded, 1)

        encoder_sum_mask = encoder_mask_expanded.sum(1)
        decoder_sum_mask = decoder_mask_expanded.sum(1)

        encoder_sum_mask = torch.clamp(encoder_sum_mask, min=1e-9)
        decoder_sum_mask = torch.clamp(decoder_sum_mask, min=1e-9)

        sentence_representation1 = encoder_sum_embeddings / encoder_sum_mask 
        sentence_representation2 = decoder_sum_embeddings / decoder_sum_mask 

        sentence_representation = torch.concat((sentence_representation1,sentence_representation2),dim=1)
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

# %%
# %%
class T5Dataset(Dataset):
    def __init__(self, args, config, tokenizer, mode, dataset_path):
        self.tokenizer = tokenizer
        self.args = args
        self.config = config

        jsonlines = json.load(open(f'{dataset_path}/{mode}_t5_{self.args.model_size}_{self.args.dataset_type}.json', 'r'))
            
        self.datas = []
        # 预处理与文本生成模型相同，保证结果公平
        random.shuffle(jsonlines)
        kept_sample_idx = 0
        for json_line in tqdm(jsonlines):
            src_txt_list = json_line['multi_doc']
            txt_gds = json_line['txt_gds'][0]
            txt_gds = tokenizer.decode(tokenizer.encode(txt_gds, max_length=self.args.max_tgt_len+2, truncation=True)[1:-1])
            relation_gds = json_line['relation_gds'][0]
            for i in range(len(src_txt_list)):
                src = tokenizer.decode(tokenizer.encode(src_txt_list[i], max_length=self.args.max_doc_len+2, truncation=True)[1:-1])
                src = '<doc-sep>' + src + '<doc-sep>'
                tgt = '<sen-sep>' + txt_gds + '<sen-sep>'
                label = 1 if i in relation_gds else 0
                self.datas.append([src, tgt, label, kept_sample_idx])  
            kept_sample_idx += 1         
            
    def __getitem__(self, index):
        src_txt, txt_gds, label, sample_idx = self.datas[index]
        labels = torch.LongTensor([label])
        inputs = self.tokenizer(src_txt, padding='max_length', max_length=self.args.max_src_len, truncation=True, return_tensors="pt")
        decoder_inputs = self.tokenizer(txt_gds, padding='max_length', max_length=self.args.max_tgt_len+20, truncation=True, return_tensors="pt")
        inputs = {k: v.squeeze(dim=0) for k ,v in inputs.items()}
        decoder_inputs = {k: v.squeeze(dim=0) for k ,v in decoder_inputs.items()}
        decoder_input_ids = decoder_inputs['input_ids']
        decoder_attention_mask = decoder_inputs['attention_mask']
        inputs.update(**{'labels':labels})
        inputs.update(**{'decoder_input_ids':decoder_input_ids})
        inputs.update(**{'decoder_attention_mask':decoder_attention_mask})
        return inputs
    def __len__(self): return len(self.datas)

# %%
# %%
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device
        labels = inputs['labels']
        outputs = model(**inputs)
        # todo: weight: 正样本权重越高， recall也高， 
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
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
            logits = outputs.logits
        return (loss, logits, labels)



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

def compute_metrics(p):
    report = {}
    preds,labels = p
    # pred_labels  = np.argmax(preds, axis=-1)
    y_true = np.squeeze(labels)
    predict = scipy.special.softmax(preds.data, 1)
    y_score = predict[:,1]
    aupr = compute_aupr(y_true, y_score)
    auc = compute_auc(y_true, y_score)
    fpr95 = compute_fpr95(y_true, y_score)
    # relation_num = sum([1 for rgt in y_true])
    # correct_num = 0
    # for i in range(len(pred_labels)):
    #     correct_num += 1 if pred_labels[i]== y_true[i] else 0
    # acc = round(float(correct_num)/float(relation_num)*100,4)
    # report['acc'] = acc
    report['aupr'] = aupr
    report['auc'] = auc
    report['fpr95'] = fpr95
    loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    loss = loss_fct(torch.tensor(preds.reshape(-1, 2)), torch.tensor(y_true.reshape(-1)))    
    report['loss'] = loss.item()   
    print('\n')
    for k ,v in report.items(): print(f'*-->{k:20}: {v}')
    print('\n')
    report = {k:v for k ,v in report.items() if not isinstance(v, dict)}
    return report



# %%
def setup_seed(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True

# %%

# %%
if __name__ == "__main__":
    
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='large', type=str)

    flags.add_argument('-dataset_path',        default='../../CODE/t5/datasets_with_generated_summary', type=str)
    flags.add_argument('-dataset_type',        default='delve_1k', type=str)
    flags.add_argument('-tokenizer_path',      default='./tokenizer_config/', type=str)
    flags.add_argument('-init_model_path',     default='', type=str)
    flags.add_argument('-decoder_layer',       default=23,    type=int)#与encoder信息融合的decoder层
    
    flags.add_argument('-max_src_len',         default=1024,    type=int)
    flags.add_argument('-max_tgt_len',         default=100,     type=int)
    flags.add_argument('-max_doc_len',         default=250,     type=int)

    flags.add_argument('-batch_size',          default=8,    type=int)
    flags.add_argument('-gradient_accumulation_steps', default=1, type=int)
    flags.add_argument('-lr',                  default=0.0001,  type=float)  
    flags.add_argument('-warmup_steps',        default=200,   type=int)
    flags.add_argument('-weight_decay',        default=0.1,  type=float)
    flags.add_argument('-epochs',              default=10,    type=int)
    flags.add_argument('-num_workers',         default=4,    type=int)
    flags.add_argument('-save_steps',          default=100,     type=int)
    flags.add_argument('-evaluation_strategy', default='epoch',     type=str)
    flags.add_argument('-eval_steps',          default=5,     type=int)
    flags.add_argument('-log_steps',           default=5,     type=int)
    flags.add_argument('-mode',                default='train', type=str)
    flags.add_argument('-output_dir',          default='./ckpts/', type=str)
    flags.add_argument('-seed',                default=2023,     type=int)
    
    
    args, unknown   = flags.parse_known_args()
    print(args.model_size)
    print(args.dataset_type, args.dataset_size)
    setup_seed(args.seed)
    
    model_size = args.model_size # 选择base或者large
    dataset_type = args.dataset_type
    tag = f'{args.dataset_type}_{args.dataset_size}_{str(args.lr)}_{args.epochs}_{args.batch_size}_{args.seed}'
    args.output_dir = f'{args.output_dir}/ft_all/{model_size}/'+tag
    args.tokenizer_path = args.tokenizer_path + model_size
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    tokenizer  = T5Tokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['<doc-sep>','<sen-sep>']})
    docsep_token_id = tokenizer.additional_special_tokens_ids[0]
    
    model_config = T5Config.from_pretrained(args.init_model_path, local_files_only=True)
    model_config.max_length = args.max_tgt_len
    model_config.decoder_layer = args.decoder_layer
    model_config.num_labels = 2
    d_model = model_config.d_model
    
    if 'train' == args.mode:
        model = T5ForSequenceClassificationNew.from_pretrained(args.init_model_path, local_files_only=True, ignore_mismatched_sizes=True, config=model_config)
        model.base_model_prefix='model'
    else:
        raise ValueError("wrong mode!")
        
    print('Reload Parameters!')
    model_dict = model.state_dict()
    checkpoint  = torch.load(f'{args.init_model_path}/'+'pytorch_model.bin', map_location = model.device)
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k.startswith('decoder') or k.startswith('encoder') or k.startswith('shared'):
            new_state_dict['model.'+k] = v
        elif k.startswith('lm'):
            pass
        else:
            new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    model.resize_token_embeddings(len(tokenizer))

    train_dataset = T5Dataset(args, config=model_config, tokenizer=tokenizer, mode='train', 
                              dataset_path=args.dataset_path)
    valid_dataset = T5Dataset(args, config=model_config, tokenizer=tokenizer, mode='valid')


    trainer_args = TrainingArguments(
            output_dir=args.output_dir,
            logging_dir='%s/log_%s' % (args.output_dir, tag),
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=int(args.batch_size/4),
            
            learning_rate=args.lr,
            lr_scheduler_type='linear', 
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
            save_strategy='epoch',
            save_total_limit=50,
            ignore_data_skip=True,
            # others
            dataloader_num_workers=args.num_workers,
            dataloader_drop_last=False,
            dataloader_pin_memory=True,
            seed=args.seed)
    trainer = CustomTrainer(
                    model          =model,
                    args           =trainer_args,
                    train_dataset  =train_dataset,
                    eval_dataset   =valid_dataset,
                    compute_metrics=compute_metrics)

    trainer.train() 



