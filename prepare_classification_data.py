from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from tqdm import tqdm
import json
from bert_input_utils import *
import numpy as np
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--bert_model_type", type=str, required=False, 
                    default="bert-base-uncased", help="the type of the bert model")
parser.add_argument("--max_length", type=int, required=False, default=256,
                        help="maximum length of the context (no. tokens)")
parser.add_argument("--context_type", type=str, required=False, 
                        default="last_turn", help="how many context to use: either last_turn or multi_turn")
args = parser.parse_args()


tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)
if args.context_type == 'last_turn':
    last_turn = True
elif args.context_type == 'multi_turn':
    last_turn = False


fusedchat_union_lexicalized_path = 'data/fusedchat_lexicalized.json'
with open(fusedchat_union_lexicalized_path, 'r') as j:
    fusedchat_union_lexicalized = json.load(j)


partition_to_id_path = 'data/partition_dictionary_partition_to_id.json'

with open(partition_to_id_path, 'r') as j:
    partition_to_id = json.load(j)

if 'npys' not in os.listdir('.'):
    os.mkdir('npys')


for partition in ['train', 'val', 'test']:
    context_ids = []
    context_masks = []
    labels = []  # chitchat: 0; TOD: 1

    for name in tqdm(partition_to_id[partition]):
        log = fusedchat_union_lexicalized[name]['log']
        history = []
        for i in range(len(log)):
            history.append(log[i]['text'])
            if i % 2 == 0: # on the user turn
                if last_turn:
                    entry = [history[-1]]
                else:
                    entry = history
                ids, masks = get_ids_and_masks_backward(entry, args.max_length, tokenizer)
                context_ids.append(ids)
                context_masks.append(masks)
                if 'dialog_act' in log[i] and 'chitchat' in log[i]['dialog_act'].keys():
                    labels.append(0)
                else:
                    labels.append(1)

    context_ids = np.array(context_ids) 
    context_masks = np.array(context_masks)
    labels = np.array(labels)

    filename_prefixes = ['context_ids', 'context_masks', 'labels']

    all_arrays = (context_ids, context_masks, labels)

    for i in range(len(filename_prefixes)):
        np.save('npys/' + filename_prefixes[i] +
                ('_last_turn_' if last_turn else '_multi_turn_') +
                partition + '.npy',
                all_arrays[i])

