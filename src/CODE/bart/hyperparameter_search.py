import os, re, csv, math, sys, time, requests,torch,random,logging,argparse,json,nltk, pickle
import shutil
import datetime
import collections
from string import punctuation
from multiprocessing.pool import Pool
from multiprocessing import Manager
import heapq
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.signal as signal
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
# from mytransformers import BartForConditionalGeneration, BartConfig, BartTokenizer
from get_best_ckpt import get_best_step
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_cross_attn_tensor(all_decoder_input_ids_and_cross_attn, output):
    _output = output.squeeze(0)
    single_output_seq_attn_series = []
    for idx in range(min(len(_output),len(all_decoder_input_ids_and_cross_attn))):
        ouput_token_id = _output[idx]
        if idx == 0:  
            continue
        decoder_input_ids_and_cross_attn = all_decoder_input_ids_and_cross_attn[idx]
        decoder_input_ids = decoder_input_ids_and_cross_attn['decoder_input_ids']
        cross_attn = [atten_weight.unsqueeze(0) for atten_weight in decoder_input_ids_and_cross_attn['all_cross_attns']]  
        cross_attn = torch.cat(cross_attn, 0) 
        cross_attn = cross_attn.transpose(0,1) 
        for batch_idx, decoder_input in enumerate(decoder_input_ids):
            if decoder_input_ids[batch_idx].equal(_output[0:idx+1]):
                single_output_seq_attn_series.append(cross_attn[batch_idx,:,:,:,:])
    return torch.cat(single_output_seq_attn_series, 2).cpu()  

def get_top2_doc(final_doc_scores):
    all_top2_doc = []
    for doc_scores in final_doc_scores:
        top2_doc_index = heapq.nlargest(2, range(len(doc_scores)), doc_scores.__getitem__)
        all_top2_doc.append([int(s) for s in sorted(top2_doc_index)])
    return all_top2_doc

def compute_accuracy(predicted_top2, top2_gd):  
    # hit@2指标
    sentence_num = sum([len(rgt[0]) for rgt in top2_gd])
    correct_num = 0
    for i in range(len(predicted_top2)):
        correct_num += len(set(predicted_top2[i][0])&set(top2_gd[i][0]))
    return round(float(correct_num)/float(sentence_num), 8)*100

def compute_fpr95(y_true, y_score):
    fpr,tpr,thres = roc_curve(y_true, y_score, pos_label=1)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    return fpr95

def attn_matrix_spilt(attn_tensor, generated_seq, tokenizer, input_txt, args):
    input_encodings = tokenizer(input_txt, padding='max_length', max_length=args.max_src_len, truncation=True, return_tensors="pt")
    generated_seq_ids = generated_seq[1:-1]  
    input_ids = input_encodings['input_ids'].squeeze(0)
    doc_sep_id = tokenizer.convert_tokens_to_ids('<doc-sep>')
    sen_sep_id = tokenizer.convert_tokens_to_ids('<sen-sep>')
    input_txt_sep_idx = torch.nonzero(input_ids == doc_sep_id).squeeze()
    generated_txt_sep_idx = torch.nonzero(generated_seq_ids == sen_sep_id).squeeze()
    input_txt_interval_tuple = [(input_txt_sep_idx[i], input_txt_sep_idx[i+1]) for i in range(len(input_txt_sep_idx)-1)]
    generated_seq_interval_tuple = [(generated_txt_sep_idx[i], generated_txt_sep_idx[i+1]) for i in range(len(generated_txt_sep_idx)-1)]
#     if len(generated_seq_interval_tuple) == 1:
#         generated_seq_interval_tuple.append((generated_seq_interval_tuple[-1][1], len(generated_seq_ids)))
    attn_tensor = attn_tensor.sum(0)  
    
    matrix_after_split = [] 
    for i in range(len(generated_seq_interval_tuple)):
        sen2all_input_doc_matrix = []
        for ii in range(len(input_txt_interval_tuple)):
            generated_sen_start_idx = generated_seq_interval_tuple[i][0] + 1
            generated_sen_end_idx = generated_seq_interval_tuple[i][1]
            input_doc_start_idx = input_txt_interval_tuple[ii][0] + 1
            input_doc_end_idx = input_txt_interval_tuple[ii][1]
            sub_matrix = attn_tensor[generated_sen_start_idx:generated_sen_end_idx, input_doc_start_idx:input_doc_end_idx]
            sub_generated_seq_ids = generated_seq_ids[generated_sen_start_idx:generated_sen_end_idx]
            sub_input_ids = input_ids[input_doc_start_idx:input_doc_end_idx]
            
            # 统一去除句号的影响
            special_tokens_to_drop = ['.', 'Ġ.']
#             special_tokens_to_drop = []
            for special_token in special_tokens_to_drop:
                special_token_id = tokenizer.convert_tokens_to_ids(special_token)  # 句号编码
                sub_matrix = sub_matrix.transpose(0,1)  # 转置
                sub_matrix = sub_matrix[sub_input_ids!=special_token_id]  # 去除query对输入txt的句号的attention系数
                sub_matrix = sub_matrix.transpose(0,1)  # 再次转置，行为generated_seq, 列为input_seq
                sub_input_ids = sub_input_ids[sub_input_ids!=special_token_id]  # 同样去除input_doc中的句号
            sen2all_input_doc_matrix.append({'sub_matrix':sub_matrix, 'sub_generated_seq_ids':sub_generated_seq_ids, 'sub_input_ids':sub_input_ids})
        
        matrix_after_split.append(sen2all_input_doc_matrix)
    return matrix_after_split


def mean_sum_pooling(attn_tensor, generated_seq, tokenizer, input_txt, args, all_ids_count):
    matrix_after_split = attn_matrix_spilt(attn_tensor, generated_seq, tokenizer, input_txt, args)
    final_scores = []
    for sen2all_input_doc_matrix in matrix_after_split:
        sen2all_doc_scores = []
        for sub_matrix_detail in sen2all_input_doc_matrix:
            sub_generated_seq_ids = sub_matrix_detail['sub_generated_seq_ids'].cpu().numpy().tolist()
            sub_matrix = sub_matrix_detail['sub_matrix']
            sub_matrix = sub_matrix.numpy()
            if args.is_medfilter:
                score_before_correction = [np.array(signal.medfilt(m,args.filter_wind)) for m in sub_matrix]
            else:
                score_before_correction = [np.array(m) for m in sub_matrix]
            score_for_correction = []
            for i in range(len(sub_generated_seq_ids)):
                score_for_correction.append([score_before_correction[i], all_ids_count[sub_generated_seq_ids[i]]])
            sen2all_doc_scores.append(score_for_correction)
        final_scores.append(sen2all_doc_scores)
    return final_scores

def word_frequency_count(attn_tensors, tokenizer):
    # 统计生成的文本中的词频，返回词为单位，以及句子为单位的统计结果
    sen_sep_id = tokenizer.convert_tokens_to_ids('<sen-sep>')
    
    all_ids = []  # 句子中token不去重统计频率
    all_sen_ids = []  # 对每个句子中出现的token去重后统计频率
    for data in attn_tensors:
        generated_seq = data['generated_seq']
        generated_seq_ids = generated_seq[1:-1]
        generated_txt_sep_idx = torch.nonzero(generated_seq_ids == sen_sep_id).squeeze()
        generated_seq_interval_tuple = [(generated_txt_sep_idx[i], generated_txt_sep_idx[i+1]) for i in range(len(generated_txt_sep_idx)-1)]
        for i in range(len(generated_seq_interval_tuple)):
            generated_sen_start_idx = generated_seq_interval_tuple[i][0] + 1
            generated_sen_end_idx = generated_seq_interval_tuple[i][1]
            sub_generated_seq_ids = generated_seq_ids[generated_sen_start_idx:generated_sen_end_idx]
            sub_generated_seq_ids_list = sub_generated_seq_ids.cpu().numpy().tolist()
            all_ids += sub_generated_seq_ids_list
            all_sen_ids += list(set(sub_generated_seq_ids_list))
    
    all_ids_count=collections.Counter(all_ids)
    all_ids_count = dict(sorted(all_ids_count.items(), key=lambda x:x[1],reverse=True))
    all_tokens_counts = {tokenizer.convert_ids_to_tokens(tid): all_ids_count[tid] for tid in all_ids_count}
    all_sen_ids_count=collections.Counter(all_sen_ids)
    all_sen_ids_count = dict(sorted(all_sen_ids_count.items(), key=lambda x:x[1],reverse=True))
    all_sen_tokens_counts = {tokenizer.convert_ids_to_tokens(tid): all_sen_ids_count[tid] for tid in all_sen_ids_count}
    return all_ids_count, all_tokens_counts, all_sen_ids_count, all_sen_tokens_counts

class BARTTestDataset(Dataset):
    def __init__(self, args, config, tokenizer, mode='train', dataset_path='./delve'):
        self.tokenizer = tokenizer
        self.args = args
        self.config = config
        
        jsonlines = json.load(open(f'./datasets_with_generated_summary/{mode}_bart_{args.model_size}_{args.dataset_type}.json', 'r'))
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
    setup_seed()
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='large', type=str)
    flags.add_argument('-dataset_path',        default='./test/', type=str)
    flags.add_argument('-dataset_type',        default='s2orc', type=str)
    flags.add_argument('-layer_num',           default=11, type=int)
    flags.add_argument('-filter_wind',         default=5, type=int)
    flags.add_argument('-is_medfilter',        default=True,  type=bool)
    flags.add_argument('-tokenizer_path',      default=f'./tokenizer_config/', type=str)
    flags.add_argument('-init_model_path',     default=f'./init_model/', type=str)
    flags.add_argument('-best_ckpt',           default='ckpt', type=str)
    flags.add_argument('-ckpts_path',          default=f'./ckpts/large/s2orc_3e-05_15_814/', type=str)
    flags.add_argument('-exp_task',            default='main_result', type=str)  # main_result, hyper-para_sen, diff_layer, diff_ckpt
    flags.add_argument('-max_src_len',         default=1024,    type=int)
    flags.add_argument('-max_tgt_len',         default=100,      type=int)
    flags.add_argument('-max_doc_len',         default=250,     type=int)
    flags.add_argument('-device',              default=0,       type=int)

    args, unknown   = flags.parse_known_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(args)

    args.tokenizer_path = f'{args.tokenizer_path}{args.model_size}/'
    args.init_model_path = f'{args.init_model_path}{args.model_size}/'

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['<doc-sep>','<sen-sep>']}) 

    if args.best_ckpt == 'init':
        model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
        ckpt_path = args.init_model_path
        best_ckpt = args.best_ckpt
        
        model = BartForConditionalGeneration.from_pretrained(ckpt_path, local_files_only=True, config=model_config).to(device)
        model.resize_token_embeddings(len(tokenizer))
    elif args.best_ckpt == 'ckpt':
        model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
        model_config.vocab_size = len(tokenizer)
        # ckpt_path = f'{args.ckpts_path}checkpoint-{args.best_ckpt}'
        
        best_ckpt = get_best_step(args.ckpts_path)
        ckpt_path = f'{args.ckpts_path}/checkpoint-{best_ckpt}'
        model = BartForConditionalGeneration.from_pretrained(ckpt_path, local_files_only=True, config=model_config).to(device)
    else:
        model_config = BartConfig.from_pretrained(args.init_model_path, local_files_only=True)
        model_config.vocab_size = len(tokenizer)
        ckpt_path = f'{args.ckpts_path}/checkpoint-{args.best_ckpt}'
        
        best_ckpt = args.best_ckpt
        # ckpt_path = f'{args.ckpts_path}/checkpoint-{best_ckpt}'
        model = BartForConditionalGeneration.from_pretrained(ckpt_path, local_files_only=True, config=model_config).to(device)
    print(ckpt_path)

    test_dataset = BARTTestDataset(args, config=model_config, tokenizer=tokenizer, mode='train', dataset_path=args.dataset_path)

    input_ids = []
    attention_masks = []
    decoder_input_ids = []
    src_txts = []
    tgt_txts = []
    all_relation_gds = []
    for data in test_dataset:
        input_ids.append(data['input_ids'])
        attention_masks.append(data['attention_mask'])
        decoder_input_ids.append(data['decoder_input_ids'])
        src_txts.append(data['src_txt'])
        tgt_txts.append(data['tgt_txt'])
        all_relation_gds.append(data['relation_gds'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        global all_case_matrix
        all_case_matrix = []
        for i in tqdm(range(input_ids.shape[0])):
            # inputs = {'input_ids': input_ids[i:i+1].to(device),'attention_mask': attention_masks[i:i+1].to(device)}
            
            # output, all_decoder_input_ids_and_cross_attn = model.generate(**inputs, num_beams=4, output_attention=True, return_dict=True, max_length=args.max_tgt_len)
            # cross_attn_tensor = get_cross_attn_tensor(all_decoder_input_ids_and_cross_attn, output)  # cross_attn_tensor: lay_num * attn_head_num * generated_seq_len * 1024; generated_seq_len指的是去除解码起始字符和终止字符（均为</s>）之后的长度

            # generated_seq = output.squeeze(0).cpu()
            
            inputs = {'input_ids': input_ids[i:i+1].to(device),'attention_mask': attention_masks[i:i+1].to(device), 'decoder_input_ids':decoder_input_ids[i].to(device)}
            output = model(**inputs, output_attentions=True, return_dict=True)
            cross_attn_tensor = output.cross_attentions
            cross_attn_tensor = [single_layer_cr[0,:,:,:] for single_layer_cr in cross_attn_tensor]
            cross_attn_tensor = cross_attn_tensor[args.layer_num].cpu()
            generated_seq = inputs['decoder_input_ids'][0].cpu()
            
            detail = {
                        'src_txt': src_txts[i],
                        'tgt_txt': tgt_txts[i], 
                        'relation_gds': all_relation_gds[i],
                        'generated_seq': generated_seq,
                        'generated_txt': tokenizer.decode(generated_seq),
                        'cross_attn_tensor': cross_attn_tensor
                    }
            all_case_matrix.append(detail)

        global all_ids_count, all_tokens_counts, all_sen_ids_count, all_sen_tokens_counts
        all_ids_count, all_tokens_counts, all_sen_ids_count, all_sen_tokens_counts = word_frequency_count(all_case_matrix, tokenizer)

        all_scores_for_para_search = []
        all_top2_gd = []
        for i in range(len(all_case_matrix)):
            attn_tensor = all_case_matrix[i]['cross_attn_tensor']
            generated_seq = all_case_matrix[i]['generated_seq']
            input_txt = all_case_matrix[i]['src_txt']
            top2_gd = all_case_matrix[i]['relation_gds']
            all_top2_gd.append(top2_gd)
            scores = mean_sum_pooling(attn_tensor, generated_seq, tokenizer, input_txt, args, all_ids_count)
            all_scores_for_para_search.append(scores)



        def hyper_para_search_worker(all_arg):
            all_scores_for_para_search = all_arg[0]
            all_top2_gd = all_arg[1]
            alpha = all_arg[2]
            beta = all_arg[3]

            all_top2_gd_and_pred_scores = []
            all_predicted_top2_doc = []
            all_top2_doc_gd = []
            y_true = []
            y_score = []
            for i in range(len(all_scores_for_para_search)):
                scores_for_para_search = all_scores_for_para_search[i]
                scores = []
                for all_sen2all_doc_scores in scores_for_para_search:
                    sen2all_doc_scores = []
                    for score_for_correction in all_sen2all_doc_scores:
                        score_after_correction = []
                        for ii in range(len(score_for_correction)):
                            softmax_s = score_for_correction[ii][0]
                            word_frequency_s = score_for_correction[ii][1]
                            score_after_correction.append(np.mean(softmax_s**alpha)/(word_frequency_s**beta))
                        sen2all_doc_scores.append(np.sum(score_after_correction))
                    scores.append(sen2all_doc_scores)
                # print(scores)
                final_predicted_top2_doc = get_top2_doc(scores)
                all_predicted_top2_doc.append(final_predicted_top2_doc)
                top2_gd = all_top2_gd[i]
                all_top2_doc_gd.append(top2_gd)

                for ii in range(len(top2_gd)):
                    gd = top2_gd[ii]
                    # gd = [int(idx) for idx in gd.split('-')]
                    raw_gd = [0,0,0,0]
                    for gd_idx in gd:
                        raw_gd[gd_idx] = 1
                    
                    score = scores[ii]
                    # print(input_txt)
                    score_sum = sum(score)
                    score = [s/score_sum for s in score]
                    for iii in range(len(raw_gd)):
                        y_true.append(raw_gd[iii])
                        y_score.append(score[iii])


            try:
                fpr95 = compute_fpr95(y_true, y_score)
                auroc = roc_auc_score(y_true, y_score)
                aupr = average_precision_score(y_true, y_score)
            except:
                print(y_score)
            
            json.dump([alpha, beta, fpr95, auroc, aupr], open(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}/{alpha}_{beta}.json', 'w'))

        all_arg = []
        for alpha in tqdm(np.arange(0,2.1,0.2)):
            for beta in np.arange(0,1.1,0.1):
                all_arg.append([all_scores_for_para_search, all_top2_gd, alpha, beta])
        
        if os.path.exists(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}'): shutil.rmtree(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}')
        os.makedirs(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}')
        with Pool(60) as pool:
            pool.map(hyper_para_search_worker, all_arg)

        all_search_result = []
        for file in tqdm(os.listdir(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}')):
                all_search_result.append(json.load(open(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}/{file}', 'r')))
        all_search_result = sorted(all_search_result, key=lambda x: float(x[2]))
        shutil.rmtree(f'./temp_hyper_para_{args.dataset_type}_{args.model_size}_{args.exp_task}')
        is_medfilter_flag = 'have_medfilter' if args.is_medfilter else 'no_medfilter'
        if not os.path.exists(f'./hyper_para_search_result_for_{args.exp_task}'): 
            os.makedirs(f'./hyper_para_search_result_for_{args.exp_task}')
        json.dump(all_search_result, open(f'./hyper_para_search_result_for_{args.exp_task}/{args.dataset_type}_{args.model_size}_ckpt-{best_ckpt}_layer{args.layer_num}_{is_medfilter_flag}_{args.filter_wind}.json', 'w'))

