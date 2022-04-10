
''' generate dialogue turns'''

import torch
from tqdm import tqdm
import os
import sys
import re
import csv
from argparse import ArgumentParser
filepath = os.path.realpath(__file__)
dirpath = os.path.dirname(filepath)
from Transformer import Transformer

APPEND_TEST_ODD_IDS = ['MUL1598','MUL2290','PMUL2636','MUL2499','SNG0991','PMUL3027','PMUL3596','MUL2376','MUL2177','PMUL1276','MUL2305','MUL2321','SNG0888','PMUL4648','PMUL2437','MUL2675','SNG0681','SNG01270','SNG01936','MUL0088','PMUL2755','MUL0264','PMUL2778','PMUL1966','MUL2119','PMUL4122','MUL2269','SNG0767','SNG0892','MUL0831','SNG0466','SNG1026','MUL2347','PMUL4077','PMUL0286','MUL2665','SNG0451','PMUL2704','PMUL4306','MUL1008']
PREPEND_TEST_ODD_IDS = ['PMUL1283','MUL1024','SNG0263','MUL1799','MUL0671','MUL2162','PMUL3126','SNG0345','PMUL4432','PMUL4255','PMUL4610','SNG01835','MUL1546','PMUL1420','PMUL1521','MUL2359','PMUL1247','SNG0391','PMUL3376','PMUL4134','PMUL4819','MUL1883','MUL2012','PMUL3737','SNG01752','MUL0515','PMUL1137','PMUL1739','PMUL1087','PMUL4884','PMUL4239','SNG01819','MUL1800','SNG02205','SNG0317','SNG02342','PMUL1424','MUL0761','SNG02240','PMUL1342']

parser = ArgumentParser()
parser.add_argument("--ckp_dir", type=str, required=False, \
                    default='runs/fusedchat_np_fused_context_delex', \
                    help="checkpoint directory")
parser.add_argument("--weights_name", type=str, required=False, \
                    default='pytorch_model.bin', \
                    help="weights_name")
parser.add_argument("--mode", type=str, required=False, \
                    default='chitchat_only', \
                    help='mode: fused, ..')
parser.add_argument("--eval_out_path", type=str, required=False, \
                    default='outs/fusedchat_fused_model_delex_context_on_fusedchat_prepend_testset.out', \
                    help='path of the evaluation output file')
parser.add_argument("--fused_chat_path", type=str, required=False, \
                    default='NeuralPipeline_DSTC8/ConvLab/tensor_cache/'
                         'fusedchat_np_tod_single_context_delex_Aug14_id_cache_string_version', \
                    help='path of the fused_chat_prepend file (string_version)')
parser.add_argument("--option", type=str, required=False, \
                    default='all_chitchat', \
                    help='only evaluate on tod-grounded chitchat or all the chitchat')
parser.add_argument("--csvfile", type=str, required=False, \
                    default='outs/fusedchat_fused_model_delex_context_on_fusedchat_prepend_testset.csv', \
                    help='csvfile')

args = parser.parse_args()
fused_chat = torch.load(args.fused_chat_path)
dataset = fused_chat
csvfile = open(args.csvfile, 'w')
csv_writer = csv.writer(csvfile, delimiter=',', quotechar='\"')
csv_writer.writerow(['ORIGINAL_ID', 'context', 'response'])
hyps = []
refs = []
model = Transformer(model_checkpoint=args.ckp_dir, mode=args.mode, weights_name=args.weights_name)
eval_out = open(args.eval_out_path, 'w')
partition = 'test'

l = []

for i in tqdm(range(len(dataset[partition]))):
    original_id = dataset[partition][i]['original_id']
    if args.option == 'all_chitchat':
        if (original_id not in APPEND_TEST_ODD_IDS) and (original_id not in PREPEND_TEST_ODD_IDS):
            continue
    elif args.option == 'tod-grounded odd':
        if (original_id not in APPEND_TEST_ODD_IDS):
            continue
    elif args.option == 'vanilla odd':
        if (original_id not in PREPEND_TEST_ODD_IDS):
            continue
    else:
        print(args.option)
        exit('unknown option')
    
    eval_out.write('original_id:')
    eval_out.write(dataset[partition][i]['original_id'])
    eval_out.write('\n')
    for j in range(len(dataset[partition][i]['utterances'])):
        # only evaluate the chitchat turns
        if dataset[partition][i]['utterances'][j]['dp'] != ['<chitchat>']:
            continue
        history = dataset[partition][i]['utterances'][j]['history']
        groundtruth_response = dataset[partition][i]['utterances'][j]['candidates'][1]
        try:
            response, conversation_mode = model.infer_fused(history)
        except:
            eval_out.write('infer failed')
            eval_out.write('\n')
            continue
        eval_out.write('history:\n')
        for turn_in_history in history:
            eval_out.write(turn_in_history)
            eval_out.write('\n')
        eval_out.write('response:')
        eval_out.write(str(response))
        eval_out.write('conversation_mode:')
        eval_out.write(str(conversation_mode))
        eval_out.write('\n')
        eval_out.write('\n')
        for k in range(len(history)):
            if k % 2 == 0:
                history[k] = 'user:' + history[k]
            else:
                history[k] = 'system:' + history[k] 
        history_one_string = '\n'.join(history)
        csv_writer.writerow([dataset[partition][i]['original_id'], \
                            '\"' + history_one_string + '\"', \
                            '\"' + str(response) + '\"'])
    eval_out.write('\n\n\n')

args_dict = vars(args)
args_dict_string = {key:str(value) for key, value in args_dict.items()}
eval_out.write('evaluation setting args: %s\n' % args_dict_string)
eval_out.close()


