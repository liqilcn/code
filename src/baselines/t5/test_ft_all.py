# %%
import os, torch, random,logging,argparse,json
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from os.path import join, exists
from torch.utils.data import Dataset
from torch.nn.functional import softmax
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers import (T5Tokenizer, 
    T5Model,
    T5PreTrainedModel,
    T5Config,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import scipy
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy import interpolate
from typing import Any, Dict, List, Optional, Set, Tuple, Union
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["WANDB_DISABLED"] = "true"

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
        # self.dense1 = nn.Linear(input_dim, inner_dim)
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
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense3(hidden_states)
        # hidden_states = torch.relu(hidden_states) 

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
    def __init__(self, args, config, tokenizer, mode='train', dataset_path='./delve', dataset_size = 'small'):
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
                # src_txt_pair = f'{src_txt_list[idx1]}[SEP]{src_txt_list[idx2]}'
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
def get_best_step(data_log):
    val_step, val_auc = [], []
    for item in data_log:
        if "eval_loss" in item.keys():
            val_auc.append(item["eval_auc"])
            val_step.append(item['step'])
    
    max_auc = max(val_auc)
    max_index = val_auc.index(max_auc)
    return val_step[max_index]


def get_metrics(ckpt_pth):
    
    file_list = os.listdir(ckpt_pth)
    ids = [int(item.split('-')[-1]) for item in file_list if 'checkpoint' in item]
    file_pth = f'checkpoint-{max(ids)}'
    
    data_log = json.load(open(ckpt_pth+'/'+file_pth+'/trainer_state.json', 'rb'))

    best_step = get_best_step(data_log["log_history"])
    test_dir = ckpt_pth + '/' + f'checkpoint-{best_step}'

    print(test_dir)

    return test_dir

# %%
if __name__ == "__main__":

    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='base', type=str)

    flags.add_argument('-dataset_path',        default='../../CODE/t5/datasets_with_generated_summary', type=str)
    flags.add_argument('-dataset_type',        default='delve_1k', type=str)
    flags.add_argument('-tokenizer_path',      default='./tokenizer_config/', type=str)
    flags.add_argument('-ckpts_path',          default='', type=str)
    flags.add_argument('-decoder_layer',       default=11,    type=int)#与encoder信息融合的decoder层

    flags.add_argument('-max_src_len',         default=1024,    type=int)
    flags.add_argument('-max_tgt_len',         default=100,     type=int)
    flags.add_argument('-max_doc_len',         default=250,     type=int)

    flags.add_argument('-batch_size',          default=16,    type=int)
    flags.add_argument('-mode',                default='test', type=str)


    args, unknown   = flags.parse_known_args()
    print(args.model_size)
    print(args.dataset_type, args.dataset_size)
    model_size = args.model_size 
    dataset_type = args.dataset_type
    args.tokenizer_path = args.tokenizer_path + model_size

    tokenizer  = T5Tokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['<doc-sep>','<sen-sep>']})
    docsep_token_id = tokenizer.additional_special_tokens_ids[0]

    model_path = get_metrics(args.ckpts_path)
    print(model_path)
    # model_path = '/home/share/qli/jupyter/train_eval_super_t5/finetune_all/ckpts/ft_all/t5_large/delve_small_0.0001_10_8_/checkpoint-1750'

    model_config = T5Config.from_pretrained(model_path, local_files_only=True)
    model_config.max_length = args.max_tgt_len
    model_config.decoder_layer = args.decoder_layer
    model_config.num_labels = 2
    d_model = model_config.d_model

    # 接下来就是进行测试，得到结果了
    test_dataset = T5Dataset(args, config=model_config, tokenizer=tokenizer, mode='test', dataset_path=args.dataset_path)

    output_detail = {
        'have_relation_score':[],
        'predicted_label':[],
        'groundtruth_label':[]
    }
    loss_fct = torch.nn.CrossEntropyLoss()  # 输入为未经过softmax的类别打分，函数内置softmax，见教程
    ckpt_path = f'{model_path}/pytorch_model.bin'
    pretrained_state = torch.load(ckpt_path)
    test_model = T5ForSequenceClassificationNew.from_pretrained(model_path, local_files_only=True, ignore_mismatched_sizes=True, config=model_config)
    test_model.load_state_dict(pretrained_state)
    test_model.cuda()
    test_model.eval()  


    with torch.no_grad():  
        for data in tqdm(test_dataset):
            inputs = data
            device = next(test_model.parameters()).device
            inputs = {k:v[None,].to(device) for k, v in inputs.items()}
            labels = inputs.get("labels")
            outputs = test_model(**inputs)
            # lnn_outputs = test_model(hidden_states=outputs, labels=labels)
            predict = torch.softmax(outputs.logits, 1).squeeze(0).cpu().numpy().tolist()
            predicted_label = int(np.argmax(predict))
            have_relation_score = predict[1]
            
            output_detail['have_relation_score'].append(have_relation_score)
            output_detail['predicted_label'].append(predicted_label)
            output_detail['groundtruth_label'].append(labels.item())


    output_detail_df = pd.DataFrame(output_detail)  

    y_ture = output_detail['groundtruth_label']
    y_score = output_detail['have_relation_score']
    aupr = compute_aupr(y_ture, y_score)
    auc = compute_auc(y_ture, y_score)
    fpr95 = compute_fpr95(y_ture, y_score)

    print({'fpr95':fpr95, 'auroc':auc, 'aupr':aupr})

    result = [round(100*fpr95, 2), round(100*auc, 2), round(100*aupr, 2)]
    seed = args.ckpts_path.split('_')[-1]
    
    if os.path.exists('./ft_all_test_results/'): os.makedirs('./ft_all_test_results/')
    with open(f'./ft_all_test_results/{model_size}_{dataset_type}_{seed}.json','w+') as file:
        file.write(json.dumps(result, indent=2, ensure_ascii=False))