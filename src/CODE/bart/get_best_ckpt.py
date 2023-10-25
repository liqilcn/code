import os
import json

def get_ckpt(ckpt_pth):
    file_list = os.listdir(ckpt_pth)
    ids = [int(item.split('-')[-1]) for item in file_list if 'checkpoint' in item]
    file_pth = f'checkpoint-{max(ids)}'
    
    data_log = json.load(open(ckpt_pth+'/'+file_pth+'/trainer_state.json', 'rb'))
    # index = get_best_step(data_log["log_history"])
    
    return data_log["log_history"]

def get_best_step(ckpt_pth):
    data_log = get_ckpt(ckpt_pth)
    
    val_step, val_loss = [], []
    for item in data_log:
        if "eval_loss" in item.keys():
            val_loss.append(item["eval_loss"])
            val_step.append(item['step'])
    
    index = val_loss.index(min(val_loss))
    return val_step[index]