 import os, re, csv, math, sys, time, requests,torch,random,logging,argparse,json,nltk, pickle
import datetime
import collections
from string import punctuation
import heapq
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from rouge_score import rouge_scorer, scoring
from old_transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from get_best_ckpt import get_best_step




class RelationBARTTestDataset(Dataset):
    def __init__(self, args, config, tokenizer, mode='test', dataset_path='./delve'):
        self.tokenizer = tokenizer
        self.args = args
        self.config = config
        if self.args.dataset_type.startswith('delve') and (mode == 'valid' or mode == 'test'):
            jsonlines = json.load(open(f'{dataset_path}{mode}_delve.json', 'r'))
        else:
            jsonlines = json.load(open(f'{dataset_path}{mode}_{self.args.dataset_type}.json', 'r'))
        self.datas = []
        for json_line in tqdm(jsonlines):
            truncated_multi_docs = []
            for ab in json_line['multi_doc']:
                truncated_multi_docs.append(tokenizer.decode(tokenizer.encode(ab, max_length=self.args.max_doc_len+2, truncation=True)[1:-1]))
            src = '<doc-sep>'.join(truncated_multi_docs)
            src = '<doc-sep>' + src + '<doc-sep>'
            truncated_txt_gds = []
            for txt_gd in json_line['txt_gds']:
                truncated_txt_gds.append(tokenizer.decode(tokenizer.encode(txt_gd, max_length=self.args.max_tgt_len+2, truncation=True)[1:-1]))
            tgt = '<sen-sep>'.join(truncated_txt_gds)
            tgt = '<sen-sep>' + tgt + '<sen-sep>'
            self.datas.append([src, tgt, json_line['relation_gds']])
            

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data_line = self.datas[index]
        src_txt, tgt_txt, relation_gds = data_line[0], data_line[1], data_line[2]
        input_encodings = self.tokenizer(src_txt, padding='max_length', max_length=self.args.max_src_len, truncation=True, return_tensors="pt")  # return_tensors="pt"不指定返回的不是tensor，是列表
        target_encodings = self.tokenizer(tgt_txt, max_length=self.args.max_tgt_len+20, truncation=True, return_tensors="pt")
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
        decoder_input_ids = target_encodings['input_ids']


        inputs = {
                'input_ids':input_ids, 
                'decoder_input_ids': decoder_input_ids,
                'attention_mask': attention_mask,
                'src_txt': src_txt,
                'tgt_txt': tgt_txt,
                'relation_gds': relation_gds
                }
        return inputs


# In[39]:


if __name__ == "__main__":
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='base', type=str)  # ,'mean_sum','mean','middle','max'
    flags.add_argument('-dataset_path',        default='../../4to1_dataset/', type=str)
    flags.add_argument('-dataset_type',        default='s2orc', type=str)
    flags.add_argument('-batch_size',          default=10, type=int)

    flags.add_argument('-tokenizer_path',      default=f'../tokenizer_config/', type=str)
    flags.add_argument('-init_model_path',     default=f'../init_model/', type=str)

    flags.add_argument('-best_ckpt',           default='assigned', type=str)  # option: 'init', 'auto', 'assigned'
    flags.add_argument('-ckpts_path',          default=f'../ckpts/base/s2orc_3e-05_15_812/', type=str)

    flags.add_argument('-max_src_len',         default=1024,    type=int)
    flags.add_argument('-max_tgt_len',         default=100,      type=int)
    flags.add_argument('-max_doc_len',         default=250,     type=int)
    flags.add_argument('-mode',                default='train',    type=str)
    flags.add_argument('-device',              default=0,       type=int)


    # args, unknown   = flags.parse_known_args()


    args, unknown   = flags.parse_known_args()
    
    if args.best_ckpt == 'auto':
        best_ckpt = get_best_step(args.ckpts_path)
        ckpt_path = f'{args.ckpts_path}/checkpoint-{best_ckpt}'
    elif args.best_ckpt == 'init':
        best_ckpt = 'init'
        ckpt_path = args.init_model_path
    elif args.best_ckpt == 'assigned':
        best_ckpt = 'assigned'
        ckpt_path = args.ckpts_path

    print(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    args.tokenizer_path = f'{args.tokenizer_path}{args.model_size}/'
    args.init_model_path = f'{args.init_model_path}{args.model_size}/'

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['<doc-sep>','<sen-sep>']}) 

    
    if args.best_ckpt == 'init':
        model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
        model_config.use_prefix = False
        model_config.preseqlen = 0

        ckpt_path = args.init_model_path
        model = BartForConditionalGeneration.from_pretrained(ckpt_path, local_files_only=True, config=model_config).to(device)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
        model_config.use_prefix = False
        model_config.preseqlen = 0
        model_config.vocab_size = len(tokenizer)

        model = BartForConditionalGeneration.from_pretrained(ckpt_path, local_files_only=True, config=model_config).to(device)
    print(ckpt_path)
    
    test_dataset = RelationBARTTestDataset(args, config=model_config, tokenizer=tokenizer, mode=args.mode, dataset_path=args.dataset_path)

    input_ids = []
    attention_masks = []
    decoder_input_ids = []
    src_txts = []
    tgt_txts = []
    for data in test_dataset:
        input_ids.append(data['input_ids'])
        attention_masks.append(data['attention_mask'])
        decoder_input_ids.append(data['decoder_input_ids'])
        src_txts.append(data['src_txt'])
        tgt_txts.append(data['tgt_txt'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        # for i in tqdm(range(input_ids.shape[0])):
        #     inputs = {'input_ids': input_ids[i*args.batch_size:i*args.batch_size+args.batch_size].to(device),'attention_mask': attention_masks[i*args.batch_size:i*args.batch_size+args.batch_size].to(device)}
        #     output = model.generate(**inputs, num_beams=4, max_length=args.max_tgt_len)
        #     generated_seq = output.squeeze(0).cpu()
        #     # generated_txt += tokenizer.batch_decode(generated_seq)

        #     string = tokenizer.decode(output[0]).replace('<pad>', '').replace('<sen-sep>', '').replace('</s>', '')
        #     sequence.append(string.strip())
            # break
        generated_txt = []
        for i in tqdm(range(math.ceil(input_ids.shape[0]/args.batch_size))):
            inputs = {'input_ids': input_ids[i*args.batch_size:i*args.batch_size+args.batch_size].to(device),'attention_mask': attention_masks[i*args.batch_size:i*args.batch_size+args.batch_size].to(device)}
            output = model.generate(**inputs, num_beams=4, max_length=args.max_tgt_len)
            generated_seq = output.squeeze(0).cpu()
            generated_txt += tokenizer.batch_decode(generated_seq)

    generated_txt = [t.replace('<sen-sep>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', '') for t in generated_txt]
    
    # 将生成的summary写入到原始数据集中，不同模型的size，type和不同的数据集生成的summary不尽相同
    if args.dataset_type.startswith('delve') and (args.mode == 'valid' or args.mode == 'test'):
        origin_dataset = json.load(open(f'{args.dataset_path}{args.mode}_delve.json', 'r'))
    else:
        origin_dataset = json.load(open(f'{args.dataset_path}{args.mode}_{self.args.dataset_type}.json', 'r'))
    
    assert len(generated_txt)== len(origin_dataset)
    if not os.path.exists('./datasets_with_generated_summary'): os.makedirs('./datasets_with_generated_summary')

    for i in range(len(generated_txt)):
        origin_dataset[i]['txt_gds'][0] = generated_txt[i].strip(', ')

    json.dump(origin_dataset, open(f'./datasets_with_generated_summary/{args.mode}_bart_{args.model_size}_{args.dataset_type}.json', 'w'))











