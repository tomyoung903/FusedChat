# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import random
import torch
import copy
from itertools import chain
from tqdm import tqdm


logger = logging.getLogger(__file__)



DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
REQUESTABLES = ['phone', 'reference', 'id', 'postcode']

slot_name = ['<leave>', '<people>', '<arrive>', '<pricerange>', '<arriveby>', '<ticket>', '<dest>', '<none>', '<leaveat>',
             '<car>', '<ref>', '<department>', '<open>', '<parking>', '<departure>', '<day>', '<type>', '<time>', '<stay>',
             '<internet>', '<phone>', '<choice>', '<destination>', '<name>', '<addr>', '<fee>', '<area>', '<post>', '<price>',
             '<depart>', '<id>', '<food>', '<stars>']

act_name = ['<restaurant-inform>', '<restaurant-recommend>', '<attraction-request>', '<hotel-request>', '<general-welcome>', '<train-offerbook>',
            '<booking-request>', '<restaurant-nooffer>', '<hospital-inform>', '<train-request>', '<train-nooffer>', '<general-bye>',
            '<hotel-select>', '<taxi-inform>', '<attraction-select>', '<attraction-nooffer>', '<booking-inform>', '<train-offerbooked>',
            '<general-greet>', '<train-inform>', '<train-select>', '<booking-nobook>', '<police-inform>', '<taxi-request>', '<attraction-inform>',
            '<restaurant-select>', '<hotel-recommend>', '<booking-book>', '<hospital-request>', '<general-reqmore>', '<restaurant-request>',
            '<hotel-nooffer>', '<hotel-inform>', '<attraction-recommend>']

dom_name = ['<hotel>', '<police>', '<restaurant>', '<train>', '<hospital>', '<taxi>', '<attraction>']

delimiters = ['<bos>', '<cs>', '<dc>', '<dp>', '<eos>', '<nm>', '<pad>', '<system>', '<user>']

requestable_slot_names_with_domain = \
    ['[attraction_addr]', '[attraction_name]', '[attraction_phone]', '[attraction_postcode]', 
     '[hospital_addr]', '[hospital_name]', '[hospital_phone]', '[hospital_postcode]',
     '[hotel_addr]', '[hotel_name]', '[hotel_phone]', '[hotel_postcode]', '[hotel_reference]', 
     '[police_addr]', '[police_name]', '[police_phone]', '[police_postcode]', 
     '[restaurant_addr]', '[restaurant_name]', '[restaurant_phone]', '[restaurant_postcode]', '[restaurant_reference]', 
     '[taxi_phone]', '[train_id]', '[train_reference]']

# the original version for only TOD
SPECIAL_TOKENS_ORIGINAL = slot_name + act_name + dom_name + delimiters + requestable_slot_names_with_domain

SPECIAL_TOKENS_chitchat_single = ['<user>', '<system>', '<pad>', '<eos>', '<bos>']


SPECIAL_TOKENS_plus_chitchat_sor = SPECIAL_TOKENS_ORIGINAL + ['<sor>'] + ['<chitchat>']



def build_input_from_segments(history, reply, tokenizer, dp=[], cs=[], lm_labels=False,
                              with_eos=True, model="gpt2",
                              mode='train', skill_mode='chitchat_single'):
    # skill_mode = chitchat_single, tod_single, chitchat_double, tod_double, classification
    bos = tokenizer.convert_tokens_to_ids('<bos>')
    eos = tokenizer.convert_tokens_to_ids('<eos>')
    user = tokenizer.convert_tokens_to_ids('<user>')
    system = tokenizer.convert_tokens_to_ids('<system>')
    cstok = tokenizer.convert_tokens_to_ids('<cs>')
    dptok = tokenizer.convert_tokens_to_ids('<dp>')
    sor = tokenizer.convert_tokens_to_ids('<sor>')
    chitchat = tokenizer.convert_tokens_to_ids('<chitchat>')

    instance = {}
    if mode == 'train':
        if skill_mode == 'chitchat_single':
            # sequence = [[bos]] + history + [[chitchat] + [system] + reply + ([eos] if with_eos else [])]
            sequence = [[bos]] + history + [[system] + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'chitchat_double':
            sequence = [[bos]] + history + [[sor] + [chitchat] + [system] + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'tod_single':
            sequence = [[bos]] + history + [[cstok] + cs + [dptok] + dp + [system] + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'tod_double':
            sequence = [[bos]] + history + [[sor] + [cstok] + cs + [dptok] + dp + [system] + reply + ([eos] if with_eos else [])]
    elif mode == 'interact':  
        if skill_mode == 'chitchat_single':
            # sequence = [[bos]] + history + [[chitchat] + reply + ([eos] if with_eos else [])]
            # <system> is NOT included in reply
            sequence = [[bos]] + history + [[system] + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'chitchat_double':
            # <system> is included in <reply>
            sequence = [[bos]] + history + [[sor] + [chitchat] + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'tod_single':
            # <system> is included in <reply> and [dptok] is included in dp
            sequence = [[bos]] + history + [[cstok] + cs + dp + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'tod_double':
            # <system> is included in <reply> and [dptok] is included in dp
            sequence = [[bos]] + history + [[sor] + [cstok] + cs + dp + reply + ([eos] if with_eos else [])]
        elif skill_mode == 'classification':
            # in the fused model, mode classification is implicitly determined by the generation of 
            # the [cstok] or the [chitchat] token
            sequence = [[bos]] + history + [[sor]]
    '''
    print('sequence1')
    for chunk in sequence:
        print([tokenizer.decode(token) for token in chunk])
        # print(sequence)
    '''
    sequence = [sequence[0]] + [[user if (len(sequence) - i) % 2 else system] + s for i, s in
                                enumerate(sequence[1:-1])] + sequence[-1:]
    '''
    print('sequence2')
    for chunk in sequence:
        print([tokenizer.decode(token) for token in chunk])
    '''
    # print('sequence2')
    # print(sequence)
    l = len([i for s in sequence for i in s])
    ctx = 1024

    if l > ctx:
        i = 1
        while l > ctx:
            d = sequence.pop(i)
            l -= len(d)
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [user if i % 2 else system for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence




def get_fusedchat(tokenizer, prepend_dataset_path, append_dataset_path, dataset_cache=None, 
    position='prepend', lexicalized_prepend_dataset_path='', lexicalized_append_dataset_path=''):
    """
        Load the FusedChat dataset.
        Args:
            tokenizer: Tokenizer to use to parse the sentence.
            prepend_dataset_path/append_dataset_path: Path of the delexicalized fusedchat dataset.
            dataset_cache: Dataset cache. If exists, this function simply loads and returns the cache.
            position: 'prepend', 'append' or 'both'.
            lexicalized_prepend_dataset_path/lexicalized_append_dataset_path: Path of the lexicalized fusedchat dataset.
    """

    # if dataset_cache already exists, load it. Otherwise save into it.
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        if position == 'prepend':
            with open(prepend_dataset_path, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
            train_dataset = dataset['train']
            valid_dataset = dataset['val']
            test_dataset = dataset['test']
            
        elif position == 'append':
            with open(append_dataset_path, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
            train_dataset = dataset['train']
            valid_dataset = dataset['val']
            test_dataset = dataset['test']

        elif position == 'both':
            with open(prepend_dataset_path, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
            train_dataset = dataset['train']
            valid_dataset = dataset['val']
            test_dataset = dataset['test']

            with open(append_dataset_path, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
            train_dataset.update(dataset['train'])
            valid_dataset.update(dataset['val'])
            test_dataset.update(dataset['test'])
        else:
            exit('The position argument should be append, prepend or both')

        def random_candidates(data):
            ind = random.choice(list(data.keys()))
            dia = [t['text'].strip() for t in data[ind]['log']]
            sys_len = len(dia)
            id_ = random.choice(range(sys_len // 2))
            return dia[2 * id_ - 1]

        def convert_act(dialog_act):
            bs = []
            sn = set()
            for d in dialog_act:
                tmp = []
                for k in list(d.keys()):
                    tmp.append('<'+k.lower()+'>')
                    sn.add('<'+k.lower()+'>')
                    for slot, value in d[k]:
                        tmp.append('<'+slot.lower()+'>')
                        tmp.append(value.lower())
                        sn.add('<'+slot.lower()+'>')
                bs.append(tmp)
            return bs, sn

        def convert_meta(dialog_meta, cur_dom):

            cs = []
            for i, d in enumerate(dialog_meta):
                dom = cur_dom[i]
                tmp = []
                if dom == 'none':
                    tmp.append('')
                else:
                    constraint = d[dom]
                    # start with the name of the domain
                    tmp.append('<'+dom.lower()+'>')
                    for b in constraint['book']:
                        # booked: booked info, such as trainID or reference number -> not logged down
                         if b != 'booked':
                             tmp.append('<' + b.lower() + '>')
                             tmp.append(constraint['book'][b])
                    for s in constraint['semi']:
                        v = constraint['semi'][s]
                        tmp.append('<'+s.lower()+'>')
                        if v in ["dont care", "don't care", "do n't care", "dontcare"]:
                            tmp.append('<dc>')
                        elif v == 'not mentioned':

                            tmp.append('<nm>')
                        else:
                            tmp.append(v)
                cs.append(' '.join(tmp))
            return cs

        def assert_version_consistency(dialog_info, dialog_info_undelexicalized):
            if len(dialog_info)!=len(dialog_info_undelexicalized):
                return False
            for i in range(len(dialog_info)):
                if len(dialog_info[i]) != len(dialog_info[i]):
                    return False
            return True

        def parse_fusedchat_data(data, undelexicalized_dataset):
            dataset = []
            doms = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'hospital', 'police']
            sns = set()
            for dia_name in tqdm(data.keys()):
                dialog_info = [t['text'].strip() for t in data[dia_name]['log']]
                if undelexicalized_dataset:
                    dialog_info_undelexicalized = [t['text'].strip() for t in undelexicalized_dataset[dia_name]['log']]
                    if not assert_version_consistency(dialog_info, dialog_info_undelexicalized):
                        print(dialog_info)
                        print(dialog_info_undelexicalized)
                        print(dia_name)
                        exit()
                dialog_act = [t['dialog_act'] for t in data[dia_name]['log']]
                # ONLY consider the system's dialog act
                dialog_act = dialog_act[1::2]
                # domain list, one for each dialog turn
                cur_dom = []
                for t in dialog_act:
                    # all dialog acts in a system turn as a big string
                    keys = [k.lower() for k in t.keys()]
                    keys = ''.join(keys)
                    # find current domain based on string matching against the dialog acts
                    # in multiwoz there is only one active domain in each turn 99% of the time
                    for d in doms:
                        if d in keys:
                            cur_dom.append(d)
                            break
                        # always duplicate the last domain in the list or add none for the first turn
                        if d == 'police':
                            if len(cur_dom) == 0:
                                cur_dom.append('none')
                            elif 'chitchat' in t.keys():
                                cur_dom.append('none')
                            else:
                                cur_dom.append(cur_dom[-1])
                dialog_meta = [t['metadata'] for t in data[dia_name]['log']]
                # ONLY consider the system's meta data
                dialog_meta = dialog_meta[1::2]
                cs = convert_meta(dialog_meta, cur_dom)  # cs: dialog state
                dp, sn = convert_act(dialog_act)  # dp: dialog act
                sns = sns.union(sn)
                dialog_len = len(dialog_info)
                if dialog_len == 0:
                    continue
                utterances = {"utterances": [], "original_id": dia_name}
                temp = {"candidates": [], "history": [], "dp": [], "cs": [], "dialog_meta": []}
                for i in range(dialog_len):
                    # in user turns
                    if i % 2 == 0:
                        if undelexicalized_dataset:
                            try:
                                temp["history"].append(dialog_info_undelexicalized[i])
                            except:
                                print(dialog_info)
                                print(dialog_info_undelexicalized)
                                exit()
                        else:
                            temp["history"].append(dialog_info[i])  # append utterances one by one as dialog history
                        temp["candidates"].append(random_candidates(data))
                        temp["candidates"].append(dialog_info[i + 1])
                        temp["dp"].append(' '.join(dp[i // 2]))
                        if cs[i // 2] != '':
                            temp["cs"].append(cs[i // 2])
                            temp['dialog_meta'].append(dialog_meta[i // 2])
                    # in system turns
                    else:
                        utterances["utterances"].append(copy.deepcopy(temp))
                        if undelexicalized_dataset:
                            temp["history"].append(dialog_info_undelexicalized[i])
                        else:
                            temp["history"].append(dialog_info[i])  # history doesn't get overwritten
                        temp["candidates"] = []
                        temp["dp"] = []
                        temp["cs"] = []
                        temp["dialog_meta"] = []
                dataset.append(utterances)
                if dia_name == 'MUL1066':
                    _ = 1
            print(list(sns))
            return dataset

        # all histories are lexicalized
        if lexicalized_prepend_dataset_path and not lexicalized_append_dataset_path:
            with open(lexicalized_prepend_dataset_path, "r", encoding="utf-8") as f:
                lexicalized_dataset = json.loads(f.read())
        elif lexicalized_append_dataset_path and not lexicalized_prepend_dataset_path:
            with open(lexicalized_append_dataset_path, "r", encoding="utf-8") as f:
                lexicalized_dataset = json.loads(f.read())
        elif lexicalized_append_dataset_path and lexicalized_prepend_dataset_path:
            with open(lexicalized_append_dataset_path, "r", encoding="utf-8") as f:
                lexicalized_append_dataset = json.loads(f.read())
            with open(lexicalized_prepend_dataset_path, "r", encoding="utf-8") as f:
                lexicalized_prepend_dataset = json.loads(f.read())
            lexicalized_append_dataset['train'].update(lexicalized_prepend_dataset['train'])
            lexicalized_append_dataset['val'].update(lexicalized_prepend_dataset['val'])
            lexicalized_append_dataset['test'].update(lexicalized_prepend_dataset['test'])
            lexicalized_dataset = lexicalized_append_dataset
        else:
            lexicalized_dataset = None

        train = parse_fusedchat_data(train_dataset, lexicalized_dataset['train'])
        valid = parse_fusedchat_data(valid_dataset, lexicalized_dataset['val'])
        test = parse_fusedchat_data(test_dataset, lexicalized_dataset['test'])
        dataset = {"train": train, "valid": valid, "test": test}
        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        if dataset_cache:
            torch.save(dataset, dataset_cache + '_string_version')
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)

    return dataset


def load_string_version_and_tokenize_dataset(dataset_cache_string_version, tokenizer):
    """
    Tokenize the dataset
    """
    logger.info("Tokenize the dataset")
    dataset = torch.load(dataset_cache_string_version)
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(dataset)
    return dataset

def get_woz_dataset_v2_4(tokenizer, dataset_path, dataset_cache=None, lexicalized_dataset_path=None):
    # dataset_cache = dataset_cache + '_' + type(tokenizer).__name__ + '_multiwoz_normalized'
    # if dataset_cache already exists, load it. Otherwise save into it.
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        train_path = os.path.join(dataset_path, 'train_v4_with_24_ds_delex_fixed.json')
        valid_path = os.path.join(dataset_path, 'val_v4_with_24_ds_delex_fixed.json')
        test_path = os.path.join(dataset_path, 'test_v4_with_24_ds_delex_fixed.json')
        with open(train_path, "r", encoding="utf-8") as f:
            train_dataset = json.loads(f.read())
        with open(valid_path, "r", encoding="utf-8") as f:
            valid_dataset = json.loads(f.read())
        with open(test_path, "r", encoding="utf-8") as f:
            test_dataset = json.loads(f.read())

        def random_candidates(data):
            ind = random.choice(list(data.keys()))
            dia = [t['text'].strip() for t in data[ind]['log']]
            sys_len = len(dia)
            id_ = random.choice(range(sys_len // 2))
            return dia[2 * id_ - 1]

        def convert_act(dialog_act):
            bs = []
            sn = set()
            for d in dialog_act:
                tmp = []
                for k in list(d.keys()):
                    tmp.append('<'+k.lower()+'>')
                    sn.add('<'+k.lower()+'>')
                    for slot, value in d[k]:
                        tmp.append('<'+slot.lower()+'>')
                        tmp.append(value.lower())
                        sn.add('<'+slot.lower()+'>')

                bs.append(tmp)
            return bs, sn

        def convert_meta(dialog_meta, cur_dom):

            cs = []
            for i, d in enumerate(dialog_meta):
                dom = cur_dom[i]
                tmp = []
                if dom == 'none':
                    tmp.append('')
                else:
                    constraint = d[dom]
                    # start with the name of the domain
                    tmp.append('<'+dom.lower()+'>')
                    for b in constraint['book']:
                        # booked: booked info, such as trainID or reference number -> not logged down
                         if b != 'booked':
                             tmp.append('<' + b.lower() + '>')
                             tmp.append(constraint['book'][b])
                    for s in constraint['semi']:
                        v = constraint['semi'][s]
                        tmp.append('<'+s.lower()+'>')
                        if v in ["dont care", "don't care", "do n't care", "dontcare"]:
                            tmp.append('<dc>')
                        elif v == 'not mentioned':
                            tmp.append('<nm>')
                        else:
                            tmp.append(v)
                cs.append(' '.join(tmp))
            return cs

        def parse_woz_data(data, undelexicalized_dataset):
            dataset = []
            doms = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'hospital', 'police']
            sns = set()
            for dia_name in tqdm(data.keys()):
                dialog_info = [t['text'].strip() for t in data[dia_name]['log']]
                if undelexicalized_dataset:
                    dialog_info_lexicalized = [t['text'].strip() for t in undelexicalized_dataset[dia_name + '.json']['log']]
                dialog_act = [t['dialog_act'] for t in data[dia_name]['log']]
                # ONLY consider the system's dialog act
                dialog_act = dialog_act[1::2]
                # domain list, one for each dialog turn
                cur_dom = []
                for t in dialog_act:
                    # all dialog acts in a system turn as a big string
                    keys = [k.lower() for k in t.keys()]
                    keys = ''.join(keys)
                    # find current domain based on string matching against the dialog acts
                    for d in doms:
                        if d in keys:
                            cur_dom.append(d)
                            break
                        # always duplicate the last domain in the list or add none for the first turn
                        if d == 'police':
                            if len(cur_dom) == 0:
                                cur_dom.append('none')
                            else:
                                cur_dom.append(cur_dom[-1])
                dialog_meta = [t['metadata'] for t in data[dia_name]['log']]
                # ONLY consider the system's meta data
                dialog_meta = dialog_meta[1::2]
                cs = convert_meta(dialog_meta, cur_dom)  # cs: dialog state
                dp, sn = convert_act(dialog_act)  # dp: dialog act
                sns = sns.union(sn)
                dialog_len = len(dialog_info)
                if dialog_len == 0:
                    continue
                utterances = {"utterances": [], "original_id": dia_name}
                temp = {"candidates": [], "history": [], "dp": [], "cs": [], "dialog_meta": []}
                for i in range(dialog_len):
                    # in user turns
                    if i % 2 == 0:
                        if undelexicalized_dataset:
                            temp['history'].append(dialog_info_lexicalized[i])
                        else:
                            temp["history"].append(dialog_info[i])  # append utterances one by one as dialog history
                        temp["candidates"].append(random_candidates(data))
                        temp["candidates"].append(dialog_info[i + 1])
                        temp["dp"].append(' '.join(dp[i // 2]))
                        if cs[i // 2] != '':
                            temp["cs"].append(cs[i // 2])
                            temp['dialog_meta'].append(dialog_meta[i // 2])
                    # in system turns
                    else:
                        utterances["utterances"].append(copy.deepcopy(temp))
                        if undelexicalized_dataset:
                            temp['history'].append(dialog_info_lexicalized[i])
                        else:
                            temp["history"].append(dialog_info[i])  # history doesn't get overwritten
                        temp["candidates"] = []
                        temp["dp"] = []
                        temp["cs"] = []
                        temp["dialog_meta"] = []
                dataset.append(utterances)
            print(list(sns))
            return dataset

        # load lexicalized data
        with open(lexicalized_dataset_path, "r", encoding="utf-8") as f:
            lexicalized_dataset = json.loads(f.read())
        
        train = parse_woz_data(train_dataset, undelexicalized_dataset = lexicalized_dataset)
        valid = parse_woz_data(valid_dataset, undelexicalized_dataset = lexicalized_dataset)
        test = parse_woz_data(test_dataset, undelexicalized_dataset = lexicalized_dataset)
        dataset = {"train": train, "valid": valid, "test": test}
        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        if dataset_cache:
            torch.save(dataset, dataset_cache + '_string_version')
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)

    return dataset

