import re, os, sys, time, requests,torch,random,logging,argparse,json,nltk
from string import punctuation
from random import shuffle
import numpy as np
from nltk import data
data.path.append('./nltk_data')
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def setup_seed(seed=2023):
     torch.manual_seed(seed)  # cpu设置随机种子
     torch.cuda.manual_seed_all(seed)  # gpu设置随机种子
     np.random.seed(seed)  # 其他的都设置随机种子
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
        
def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = calculate_rouge(output_lns=decoded_preds, reference_lns=decoded_labels, use_stemmer=True)
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}
    

class BARTDataset(Dataset):
    def __init__(self, args, config, tokenizer, mode='train', dataset_path='./delve'):
        self.tokenizer = tokenizer
        self.args = args
        self.config = config
        jsonlines = json.load(open(f'{dataset_path}/{mode}_pretrain_{self.args.dataset_type}.json', 'r'))
        self.datas = []
        for json_line in tqdm(jsonlines):
            truncated_multi_docs = []
            for ab in json_line['multi_doc']:
                truncated_multi_docs.append(tokenizer.decode(tokenizer.encode(ab, max_length=self.args.max_doc_len+2, truncation=True)[1:-1]))
            truncated_txt_gds = []
            for txt_gd in json_line['txt_gds']:
                truncated_txt_gds.append(tokenizer.decode(tokenizer.encode(txt_gd, max_length=self.args.max_tgt_len+2, truncation=True)[1:-1]))
            src = '<doc-sep>'.join(truncated_multi_docs)
            src = '<doc-sep>' + src + '<doc-sep>'
            tgt = '<sen-sep>'.join(truncated_txt_gds)
            tgt = '<sen-sep>' + tgt + '<sen-sep>'
            self.datas.append([src, tgt])
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data_line = self.datas[index]
        src_txt, tgt_txt = data_line[0], data_line[1]
        input_encodings = self.tokenizer(src_txt, padding='max_length', max_length=self.args.max_src_len, truncation=True, return_tensors="pt")  # return_tensors="pt"不指定返回的不是tensor，是列表
        target_encodings = self.tokenizer(tgt_txt, padding='max_length', max_length=self.args.max_tgt_len, truncation=True, return_tensors="pt")
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = input_encodings['input_ids'].squeeze(0)
        attention_mask = input_encodings['attention_mask'].squeeze(0)
        # labels (torch.LongTensor of shape (batch_size, sequence_length), optional) — Labels for computing the masked language modeling loss. Indices should either be in [0, ..., config.vocab_size] or -100 (see input_ids docstring). Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size].
        tgt_inputs_ids = target_encodings['input_ids']  # <s>i love you</s>，<s>和</s>分别为解码起始符和终止符
        decoder_input_ids = tgt_inputs_ids[:, :-1].contiguous().squeeze(0)  # 第0维是batch，第1维舍弃最后一位，即</s>， 例如<s>i love you
        labels = tgt_inputs_ids[:, 1:].clone()  # 模型学习的groundtruth，去掉第一个字符，即：i love you</s>
        labels[tgt_inputs_ids[:, 1:] == pad_token_id] = -100
        
        
        inputs = {
                'input_ids':input_ids, 
                'attention_mask': attention_mask, 
                'labels': labels.squeeze(0), 
                'decoder_input_ids': decoder_input_ids
                }
        return inputs

if __name__ == "__main__":
    setup_seed()
    flags = argparse.ArgumentParser()
    flags.add_argument('-model_size',          default='base', type=str)

    flags.add_argument('-dataset_path',        default='./4to1_dataset', type=str)
    flags.add_argument('-dataset_type',        default='delve', type=str)
    flags.add_argument('-tokenizer_path',      default='./tokenizer_config/', type=str)
    flags.add_argument('-init_model_path',     default='./init_model/', type=str)
    
    
    flags.add_argument('-max_src_len',         default=1024,    type=int)
    flags.add_argument('-max_tgt_len',         default=100,     type=int)
    flags.add_argument('-max_doc_len',         default=250,     type=int)

    flags.add_argument('-batch_size',          default=8,    type=int)
    flags.add_argument('-gradient_accumulation_steps', default=1, type=int)
    flags.add_argument('-lr',                  default=3e-5,  type=float)  # BART 3e-5 达到最佳
    flags.add_argument('-warmup_steps',        default=200,   type=int)
    flags.add_argument('-weight_decay',        default=0.1,  type=float)
    flags.add_argument('-epochs',              default=15,    type=int)
    flags.add_argument('-num_workers',         default=4,    type=int)
    flags.add_argument('-save_steps',          default=100,     type=int)
    flags.add_argument('-evaluation_strategy', default='epoch',     type=str)
    flags.add_argument('-eval_steps',          default=5,     type=int)
    flags.add_argument('-log_steps',           default=5,     type=int)
    flags.add_argument('-mode',                default='train', type=str)
    flags.add_argument('-output_dir',          default='./ckpts/', type=str)
    
    
    args, unknown   = flags.parse_known_args()
    print(args.model_size)
    print(args.dataset_type)
    model_size = args.model_size # 选择base或者large
    tag = f'{args.dataset_type}_{str(args.lr)}_{args.epochs}_{args.batch_size}{args.gradient_accumulation_steps}{args.num_workers}'
    args.output_dir = f'{args.output_dir}{model_size}/'+tag
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    tokenizer = BartTokenizer.from_pretrained(f'{args.tokenizer_path}{model_size}/', local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['<doc-sep>','<sen-sep>']})
    
    model_config = BartConfig.from_pretrained(f'{args.init_model_path}{model_size}/', local_files_only=True)
    model_config.max_length = args.max_tgt_len
    
    model = BartForConditionalGeneration.from_pretrained(f'{args.init_model_path}{model_size}/', local_files_only=True, ignore_mismatched_sizes=True, config=model_config)
    
    model.resize_token_embeddings(len(tokenizer))
    
        
    train_dataset = BARTDataset(args, config=model_config, tokenizer=tokenizer, mode='train', dataset_path=args.dataset_path)
    valid_dataset = BARTDataset(args, config=model_config, tokenizer=tokenizer, mode='valid', dataset_path=args.dataset_path)
    
    trainer_args = Seq2SeqTrainingArguments(
        # no_cuda=False,
        output_dir=args.output_dir,
        logging_dir='%s/log_%s' % (args.output_dir, tag),
        overwrite_output_dir=False,
        # train config
        # label_names=TARGE_NAMES, # if set variable 'label_names', compute_metric will not run;
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # max_steps=200,
        # eval
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.batch_size,
        # optimizer
        learning_rate=args.lr,
        lr_scheduler_type='linear', # "linear, cosine, cosine_with_restarts, polynomial, constant_with_warmup"
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
        save_total_limit=30,
        # load_best_model_at_end=True,
        # metric_for_best_model='eval_f1',
        ignore_data_skip=True,
        # others
        predict_with_generate=True,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=False,
        dataloader_pin_memory=True,
        seed=2023)
    
    trainer = Seq2SeqTrainer(
                  model          =model,
                  args           =trainer_args,
                  train_dataset  =train_dataset,
                  eval_dataset   =valid_dataset,
                  tokenizer      =tokenizer,
                  compute_metrics=compute_metrics)  
    
    trainer.train()