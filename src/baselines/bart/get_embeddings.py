# %%
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DISABLED"] = "true"

# %%
def setup_seed(seed=1996):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True

class BartExtractEembedding(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.decoder_layer = config.decoder_layer
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
    ) :
       
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
        return sentence_representation



# %%

class BARTDataset(Dataset):
    def __init__(self, args, config, tokenizer, mode='train', dataset_path='./delve', dataset_size = 'small'):
        self.tokenizer = tokenizer
        self.args = args
        self.config = config
        
        jsonlines = json.load(open(f'{dataset_path}/{mode}_bart_{self.args.model_size}_{self.args.dataset_type}.json', 'r'))
        
        self.datas = []
        # 预处理与文本生成模型相同，保证结果公平
        random.shuffle(jsonlines)
        kept_sample_idx = 0
        for json_line in tqdm(jsonlines):
            src_txt_list = json_line['multi_doc']
            txt_gds = json_line['txt_gds'][0]
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
        decoder_inputs = self.tokenizer(txt_gds, padding='max_length', max_length=self.args.max_tgt_len, truncation=True, return_tensors="pt")
        inputs = {k: v.squeeze(dim=0) for k ,v in inputs.items()}
        decoder_inputs = {k: v.squeeze(dim=0) for k ,v in decoder_inputs.items()}
        decoder_input_ids = decoder_inputs['input_ids']
        decoder_attention_mask = decoder_inputs['attention_mask']
        inputs.update(**{'labels':labels})
        inputs.update(**{'decoder_input_ids':decoder_input_ids})
        inputs.update(**{'decoder_attention_mask':decoder_attention_mask})
        return inputs, sample_idx
    def __len__(self): return len(self.datas)

class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    def __getitem__(self, index):
        return {'hidden_states':self.data[index][0],'labels':self.data[index][1]}
    def __len__(self): 
        return len(self.data)



if __name__ == "__main__":
    setup_seed()
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='large', type=str)

    flags.add_argument('-dataset_path',        default='../../CODE/t5/datasets_with_generated_summary', type=str)
    flags.add_argument('-dataset_type',        default='delve_1k', type=str)
    flags.add_argument('-tokenizer_path',      default='./tokenizer_config/', type=str)
    flags.add_argument('-init_model_path',     default='', type=str)

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
    flags.add_argument('-eval_steps',          default=5,     type=int)
    flags.add_argument('-log_steps',           default=5,     type=int)
    flags.add_argument('-mode',                default='train', type=str)

    # %%
    args, unknown   = flags.parse_known_args()
    print(args.model_size)
    print(args.dataset_type)
    model_size = args.model_size # 选择base或者large
    dataset_type = args.dataset_type
    args.tokenizer_path = args.tokenizer_path + model_size

    tokenizer  = BartTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['<doc-sep>','<sen-sep>']})
    docsep_token_id = tokenizer.additional_special_tokens_ids[0]

    model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
    model_config.max_length = args.max_tgt_len
    model_config.decoder_layer = args.decoder_layer
    model_config.num_labels = 2
    d_model = model_config.d_model


    # %%
    if 'train' == args.mode:
        model = BartExtractEembedding.from_pretrained(args.init_model_path, local_files_only=True, ignore_mismatched_sizes=True, config=model_config)
        model.cuda()
    else:
        raise ValueError("wrong mode!")

    # %%
    model.resize_token_embeddings(len(tokenizer))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    # for name, parameter in model.named_parameters():
    #     print(name,parameter.requires_grad)
        
    embed_path = f'./embedding/{model_size}/{dataset_type}'
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)

    # %%
    train_dataset = BARTDataset(args, config=model_config, tokenizer=tokenizer, mode='train', \
    dataset_path=args.dataset_path)
    train_embedding_lst = []
    for data in tqdm(train_dataset):
        inputs, sample_index = data
        device = next(model.parameters()).device
        inputs = {k:v[None,].to(device) for k, v in inputs.items()}
        labels = inputs.get("labels")
        outputs = model(**inputs)
        train_embedding_lst.append((outputs[0].to('cpu'),labels[0].to('cpu')))
    train_emb_dataset = EmbeddingDataset(train_embedding_lst)
    pkl.dump(train_emb_dataset, open(f'{embed_path}/train.pkl', 'wb'))
    print('train embedding is saved')

    # %%
    valid_dataset = BARTDataset(args, config=model_config, tokenizer=tokenizer, mode='valid',\
        dataset_path=args.dataset_path)

    valid_embedding_lst = []
    for data in tqdm(valid_dataset):
        inputs, sample_index = data
        device = next(model.parameters()).device
        inputs = {k:v[None,].to(device) for k, v in inputs.items()}
        labels = inputs.get("labels")
        outputs = model(**inputs)
        valid_embedding_lst.append((outputs[0].to('cpu'),labels[0].to('cpu')))
    valid_emb_dataset = EmbeddingDataset(valid_embedding_lst)
    pkl.dump(valid_emb_dataset, open(f'{embed_path}/valid.pkl', 'wb'))
    print('valid embedding is saved')

    # %%
    test_dataset = BARTDataset(args, config=model_config, tokenizer=tokenizer, mode='test',\
        dataset_path=args.dataset_path)

    test_embedding_lst = []
    for data in tqdm(test_dataset):
        inputs, sample_index = data
        device = next(model.parameters()).device
        inputs = {k:v[None,].to(device) for k, v in inputs.items()}
        labels = inputs.get("labels")
        outputs = model(**inputs)
        test_embedding_lst.append((outputs[0].to('cpu'),labels[0].to('cpu')))
    test_emb_dataset = EmbeddingDataset(test_embedding_lst)

    pkl.dump(test_emb_dataset, open(f'{embed_path}/test.pkl', 'wb'))
    print('test embedding is saved')




