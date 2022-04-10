import random
import torch
import copy
import torch.nn.functional as F
import logging
import sys
import conversation_mode_classification as cmc
from pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer
from dbquery import query_fuzzy_and_normalized

from util import build_input_from_segments, SPECIAL_TOKENS_ORIGINAL, \
    SPECIAL_TOKENS_plus_chitchat_sor, SPECIAL_TOKENS_chitchat_single, act_name, slot_name
DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"


class Generator():
    def __init__(self,
                 model_checkpoint='',
                 max_history=15,
                 device='cuda',
                 no_sample=False,
                 max_length=40,
                 min_length=1,
                 seed=42,
                 temperature=0.9,
                 top_k=0,
                 top_p=0.8,
                 mode='fused',
                 weights_name='pytorch_model.bin',
                 log_dir ='outs/temp.log'):
        self.logger = logging.getLogger(log_dir)
        self.logger.setLevel(logging.WARNING)
        fh = logging.FileHandler(log_dir)          
        fh.setLevel(logging.WARNING)
        self.logger.addHandler(fh)
        self.mode = mode
        self.model_checkpoint = model_checkpoint
        self.max_history = max_history
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.no_sample = no_sample
        self.device = device
        self.seed = seed
        self.domains = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'police', 'hospital']
        self.cs_mapping = {'restaurant': ['food', 'pricerange', 'name', 'area'],
                           'hospital': ['department', 'phone'],
                           'hotel': ['name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type'],
                           'attraction': ['type', 'name', 'area'],
                           'train': ['leaveat', 'destination', 'day', 'arriveby', 'departure'],
                           'taxi': ['leaveat', 'destination', 'departure', 'arriveby'],
                           'police': []}                           
        dia_act = open('data/dialog_act_slot.txt', 'r')
        f = dia_act.read().split('\n')
        self.dia_act_dict = {}
        key = ""
        for i, c in enumerate(f):
            if i == 0:
                continue  # User Dialog Act case
            t = c.split('\t')
            if len(t) == 1:
                key = t[0].lower()
                self.dia_act_dict[key] = []
            else:
                self.dia_act_dict[key].append(t[-1].strip().lower())
        self.logger.info('self.dia_act_dict')
        self.logger.info(self.dia_act_dict)
        random.seed(self.seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.cur_dom = ''
        self.prev_dom = ''
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2DoubleHeadsModel    
        self.logger.info('self.model_checkpoint')
        self.logger.info(self.model_checkpoint)
        self.model, loading_info = model_class.from_pretrained(self.model_checkpoint, weights_name=weights_name)
        self.logger.info('loading_info:')
        self.logger.info(loading_info)
        if mode == 'chitchat_single':
            self.SPECIAL_TOKENS = SPECIAL_TOKENS_chitchat_single
        elif mode == 'tod_single':
            self.SPECIAL_TOKENS = SPECIAL_TOKENS_ORIGINAL
        elif mode == 'fused':
            self.SPECIAL_TOKENS = SPECIAL_TOKENS_plus_chitchat_sor
        else:
            exit('mode is unknown')
        self.tokenizer = tokenizer_class.from_pretrained(model_checkpoint, unk_token='<|unkwn|>')
        SPECIAL_TOKENS_DICT = {}
        for st in self.SPECIAL_TOKENS:
            SPECIAL_TOKENS_DICT[st] = st
        self.logger.info(model_checkpoint)
        self.logger.info(self.tokenizer)
        self.logger.info(len(self.tokenizer))
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        self.logger.info(len(self.tokenizer))
        self.logger.info(self.model)
        self.model.to(self.device)
        self.model.eval()
        self.count = 0
        self.reset()

    def sample_sequence(self, history, current_output=None, mode=None):
        self.logger.info('history:')
        self.logger.info(history)
        if mode == 'chitchat_single':
            return self.sample_sequence_chitchat(history, current_output)
        elif mode == 'tod_single':
            return self.sample_sequence_tod(history, current_output)
        elif mode == 'fused':
            return self.sample_sequence_fused(history, current_output)

    def sample_sequence_chitchat(self, history, current_output=None):
        eos = [self.tokenizer.convert_tokens_to_ids('<eos>')]
        if current_output is None:
            current_output = []

        cs_dict = {}
        kb_results = {}
        i = 0
        dp = []
        cs = []
        whole_kb = None
        while i < self.max_length:
            instance, _ = build_input_from_segments(history, current_output, self.tokenizer,
                                                           skill_mode='chitchat_single', with_eos=False,
                                                           mode='interact')
            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)
            
            logits, _, _ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits)
            probs = F.softmax(logits, dim=-1)
            prev = torch.topk(probs, 1)[1]
            self.logger.info(probs)
            self.logger.info(prev)
            prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)
            if i < self.min_length and prev.item() in eos:
                b = 0
                while prev.item() in eos:
                    if b == 3:
                        break
                    prev = torch.multinomial(probs, num_samples=1)
                    b += 1

            if prev.item() in eos:
                break
            current_output.append(prev.item())
            i += 1

        return current_output, dp, cs_dict, kb_results, whole_kb

    def sample_sequence_tod(self, history, current_output=None):
        dptok = [self.tokenizer.convert_tokens_to_ids('<dp>')]
        sys = [self.tokenizer.convert_tokens_to_ids('<system>')] 
        eos = [self.tokenizer.convert_tokens_to_ids('<eos>')]

        if current_output is None:
            current_output = []
        cs_dict = {}
        kb_results = {}
        i = 0
        dp_count = 0
        cs_count = 0
        dp = []
        cs = []
        cs_done = 0
        dp_done = 0
        constraints = []
        whole_kb = None
        while i < self.max_length:
            instance, _ = build_input_from_segments(history, current_output, self.tokenizer, dp=dp, cs=cs,
                                                           with_eos=False, mode='interact', skill_mode='tod_single')
            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)
            
            logits, _, _ = self.model(input_ids, token_type_ids=token_type_ids)

            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits)
            probs = F.softmax(logits, dim=-1)
            if not dp_done:
                prev = torch.topk(probs, 1)[1]
            else:
                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

            if i < self.min_length and prev.item() in eos:
                b = 0
                while prev.item() in eos:
                    if b == 3:
                        break
                    prev = torch.multinomial(probs, num_samples=1)
                    b += 1
            # self.logger.info('mode type:')
            # self.logger.info(self.mode)
            # self.logger.info('test token issue')
            # self.logger.info(self.tokenizer.decode(prev.item()))
            if prev.item() in eos:
                break

            if prev.item() in dptok:
                if cs_count == 0:
                    self.logger.info('cs: ')
                    self.logger.info(cs)
                    cs_text = self.decode(cs).strip()
                    self.logger.info('cs_text: ' + cs_text)
                    # update the domain
                    if self.cur_dom != cs_text.split(' ')[0][1:-1] and cs_text.split(' ')[0][1:-1] in self.domains:
                        self.cur_dom = cs_text.split(' ')[0][1:-1]
                    self.logger.info('self.cur_dom: ' + self.cur_dom)
                    # keys are the informable slot names
                    keys = self.cs_mapping[self.cur_dom] if self.cur_dom else []
                    # build the cs_dict
                    if keys != []:
                        prev_key = (0, '')
                        cs_tok = cs_text.split(' ')
                        self.logger.info('cs_tok: ')
                        self.logger.info(cs_tok)
                        for j, tok in enumerate(cs_tok):
                            if tok[1:-1] in keys:
                                if prev_key[1] != '':
                                    cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1: j])
                                prev_key = (j, tok[1:-1])
                            if j == len(cs_tok) - 1:
                                cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1:])
                    self.logger.info('cs_dict: ')
                    self.logger.info(cs_dict)

                    constraints = []
                    cs_key = []
                    # construct constraints for kb search
                    for k in cs_dict:
                        if not cs_dict[k] in ['<nm>', '', '<nm> ']:
                            if cs_dict[k] in ['<dc>', '<dc> ']:
                                if k == 'arriveby':
                                    constraints.append(['arriveBy', 'dontcare'])
                                elif k == 'leaveat':
                                    constraints.append(['leaveAt', 'dontcare'])
                                else:
                                    constraints.append([k, 'dontcare'])
                            else:
                                if k == 'arriveby':
                                    constraints.append(['arriveBy', cs_dict[k]])
                                elif k == 'leaveat':
                                    constraints.append(['leaveAt', cs_dict[k]])
                                else:
                                    constraints.append([k, cs_dict[k]])
                            cs_key.append(k)
                    self.logger.info('constraints: ')
                    self.logger.info(constraints)
                    kb_results = query_fuzzy_and_normalized(self.cur_dom, constraints) if self.cur_dom else None
                    self.logger.info('kb_results: ')
                    self.logger.info(kb_results)
                    if self.cur_dom == 'train':
                        # sort the kb results for time-related slot names
                        if 'leaveat' in cs_key:
                            kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                        elif 'arriveby' in cs_key:
                            kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)

                    whole_kb = kb_results
                    kb_results = self.convert_kb(kb_results[0]) if kb_results else None
                    self.logger.info('kb_results after convert_kb: ')
                    self.logger.info(kb_results)
                    cs_count += 1
                    cs_done += 1
                    i = 0

            if prev.item() in sys:
                if dp_count == 0:
                    self.logger.info('dp: ')
                    self.logger.info(dp)
                    dialog_act = dp[1:]
                    da_text = self.decode(dialog_act).strip()
                    self.logger.info('da_text: ')
                    self.logger.info(da_text)
                    da_tok = da_text.split(' ')
                    toks = []
                    for i, t in enumerate(da_tok):
                        if t in act_name:
                            toks.extend(t[1:-1].split('-'))
                        elif t in slot_name:
                            toks.append(t[1:-1])
                        else:
                            toks.append(t)
                    self.logger.info('toks: ')
                    self.logger.info(toks)
                    da_dict = self.convert_da(' '.join(toks), self.dia_act_dict)
                    self.logger.info('da_dict after convert_da')
                    self.logger.info(da_dict)
                    da_dict = self.convert_value(da_dict, constraints, kb_results, whole_kb)
                    self.logger.info('da_dict after convert_value:')
                    self.logger.info(da_dict)
                    bs = []
                    for d in da_dict:
                        bs.append('<' + d.lower() + '>')
                        for slot, value in da_dict[d]:
                            bs.append('<' + slot.lower() + '>')
                            if isinstance(value, dict):
                                for k in value.keys():
                                    bs.append(k)
                                    bs.append(value[k])
                            else:
                                bs.append(value.lower())
                    dp = self.tokenizer.encode('<dp> ' + ' '.join(bs))
                    i = 0
                    dp_count += 1
                    dp_done += 1

            if not cs_done:
                cs.append(prev.item())
            elif not dp_done:
                dp.append(prev.item())
            else:
                current_output.append(prev.item())
            i += 1
        self.prev_dom = self.cur_dom
        self.logger.info('current_output:')
        self.logger.info(current_output)
        return current_output[1:], dp[1:], cs_dict, kb_results, whole_kb

    def sample_sequence_fused(self, history, current_output=None):        
        cstok = [self.tokenizer.convert_tokens_to_ids('<cs>')]
        chitchat = [self.tokenizer.convert_tokens_to_ids('<chitchat>')]
        dptok = [self.tokenizer.convert_tokens_to_ids('<dp>')]
        sys = [self.tokenizer.convert_tokens_to_ids('<system>')] 
        eos = [self.tokenizer.convert_tokens_to_ids('<eos>')]

        if current_output is None:
            current_output = []
        cs_dict = {}
        kb_results = {}
        i = 0
        dp_count = 0
        cs_count = 0
        dp = []
        cs = []
        
        # in the fused model, mode classification is implicitly determined by the generation of 
        # the [cstok] or the [chitchat] token
        skill_mode = 'classification'
        conversation_skill_mode = 'tod'
        cs_done = 0
        dp_done = 0
        constraints = []
        whole_kb = None
        while i < self.max_length:
            instance, _ = build_input_from_segments(history, current_output, self.tokenizer, dp=dp, cs=cs,
                                                           with_eos=False, mode='interact', skill_mode=skill_mode)
            
            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)
            logits, _, _ = self.model(input_ids, token_type_ids=token_type_ids)
            
            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits)
            probs = F.softmax(logits, dim=-1)
            prev = torch.topk(probs, 1)[1]
            if skill_mode == 'classification':
                if prev.item() in chitchat:
                    skill_mode = 'chitchat_double'
                    continue
                elif prev.item() in cstok:
                    skill_mode = 'tod_double'
                    self.logger.info('skill_mode is ' + skill_mode)
                    continue
                else:
                    self.logger.info('The first generated token is neither <cs> nor <chitchat>.')
                    self.logger.info(probs)
                    self.logger.info(prev)
                    self.logger.info(self.tokenizer.decode(prev.item()))
                    skill_mode = 'tod_double'
                    self.logger.info('skill_mode is ' + skill_mode)
                    continue

            if skill_mode == 'chitchat_double':
                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)
                if i < self.min_length and prev.item() in eos:
                    b = 0
                    while prev.item() in eos:
                        if b == 3:
                            break
                        prev = torch.multinomial(probs, num_samples=1)
                        b += 1

                if prev.item() in eos:
                    break
                current_output.append(prev.item())  # include <system>

            if skill_mode == 'tod_double':
                if not dp_done:
                    prev = torch.topk(probs, 1)[1]
                else:
                    # do some sampling when generating the response
                    prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

                # if eos is produced before min_length is reached. Give it 3 more tries to find a non-eos.
                if i < self.min_length and prev.item() in eos:
                    b = 0
                    while prev.item() in eos:
                        if b == 3:
                            break
                        prev = torch.multinomial(probs, num_samples=1)
                        b += 1
                # self.logger.info('test token issue')
                # self.logger.info(self.tokenizer.decode(prev.item()))
                if self.tokenizer.decode(prev.item()) == '<chitchat>':
                    conversation_skill_mode = 'chitchat'
                if prev.item() in eos:
                    break

                if prev.item() in dptok:
                    if cs_count == 0:
                        cs_text = self.decode(cs).strip()
                        # cur_dom is predicted as part of cs_text
                        if self.cur_dom != cs_text.split(' ')[0][1:-1] and cs_text.split(' ')[0][1:-1] in self.domains:
                            self.cur_dom = cs_text.split(' ')[0][1:-1]

                        keys = self.cs_mapping[self.cur_dom] if self.cur_dom else []

                        if keys != []:

                            prev_key = (0, '')
                            cs_tok = cs_text.split(' ')
                            for j, tok in enumerate(cs_tok):
                                if tok[1:-1] in keys:
                                    if prev_key[1] != '':
                                        cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1: j])
                                    prev_key = (j, tok[1:-1])
                                if j == len(cs_tok) - 1:
                                    cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1:])

                        constraints = []
                        cs_key = []
                        for k in cs_dict:
                            if not cs_dict[k] in ['<nm>', '', '<nm> ']:
                                if cs_dict[k] in ['<dc>', '<dc> ']:
                                    if k == 'arriveby':
                                        constraints.append(['arriveBy', 'dontcare'])

                                    elif k == 'leaveat':
                                        constraints.append(['leaveAt', 'dontcare'])
                                    else:
                                        constraints.append([k, 'dontcare'])
                                else:
                                    if k == 'arriveby':
                                        constraints.append(['arriveBy', cs_dict[k]])

                                    elif k == 'leaveat':
                                        constraints.append(['leaveAt', cs_dict[k]])
                                    else:
                                        constraints.append([k, cs_dict[k]])
                                cs_key.append(k)

                        kb_results = query_fuzzy_and_normalized(self.cur_dom, constraints) if self.cur_dom else None

                        if self.cur_dom == 'train':
                            if 'leaveat' in cs_key:
                                kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                            elif 'arriveby' in cs_key:
                                kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)

                        whole_kb = kb_results
                        kb_results = self.convert_kb(kb_results[0]) if kb_results else None
                        cs_count += 1
                        cs_done += 1
                        i = 0

                if prev.item() in sys:

                    if dp_count == 0:
                        dialog_act = dp[1:]
                        da_text = self.decode(dialog_act).strip()

                        da_tok = da_text.split(' ')
                        toks = []
                        for i, t in enumerate(da_tok):

                            if t in act_name:
                                toks.extend(t[1:-1].split('-'))
                            elif t in slot_name:
                                toks.append(t[1:-1])
                            else:
                                toks.append(t)
                        da_dict = self.convert_da(' '.join(toks), self.dia_act_dict)
                        da_dict = self.convert_value(da_dict, constraints, kb_results, whole_kb)
                        bs = []

                        for d in da_dict:
                            bs.append('<' + d.lower() + '>')
                            for slot, value in da_dict[d]:
                                bs.append('<' + slot.lower() + '>')
                                if isinstance(value, dict):
                                    for k in value.keys():
                                        bs.append(k)
                                        bs.append(value[k])
                                else:
                                    bs.append(value.lower())
                        dp = self.tokenizer.encode('<dp> ' + ' '.join(bs))
                        i = 0
                        dp_count += 1
                        dp_done += 1

                if not cs_done:
                    cs.append(prev.item())
                elif not dp_done:
                    dp.append(prev.item())
                else:
                    current_output.append(prev.item())
                self.prev_dom = self.cur_dom

            i += 1
        self.logger.info(self.tokenizer.decode(current_output[1:]))
        if skill_mode == 'chitchat_double':
            return current_output[1:], dp, cs_dict, kb_results, whole_kb, conversation_skill_mode
        else:
            return current_output[1:], dp[1:], cs_dict, kb_results, whole_kb, conversation_skill_mode

    def convert_da(self, da, dia_act_dict):
        ''' Convert  '''
        da = da.replace('i d', 'id')
        da_list = da.split(' ')
        # parking + none is force converted to parking + yes
        for p in range(len(da_list)):
            if p != len(da_list) - 1 and da_list[p] == 'parking' and da_list[p + 1] == 'none':
                da_list[p + 1] = 'yes'
        i = 0
        # find the dialog act names (e.g., train-inform)
        idlist = []
        while i < len(da_list):
            act = '-'.join(da_list[i:i + 2])
            if act in dia_act_dict.keys():
                idlist.append(i)
            i += 1
        da_dict = {}
        # for each slot act, find the slot name and value
        for i in range(len(idlist)):
            act = '-'.join(da_list[idlist[i]:idlist[i] + 2])

            if i == len(idlist) - 1:
                sv = da_list[idlist[i] + 2:]
            else:
                sv = da_list[idlist[i] + 2:idlist[i + 1]]
            sv_id = []
            for slot in dia_act_dict[act]:
                for j in range(len(sv)):
                    if slot == sv[j]:
                        if j > 0 and sv[j - 1] != 'none':
                            sv_id.append(j)
                        if j == 0:
                            sv_id.append(j)
            sv_list = []
            sv_id.sort()
            k = 0
            while k < len(sv_id):
                if k == len(sv_id) - 1:
                    sv_list.append([sv[sv_id[k]], ' '.join(sv[sv_id[k] + 1:])])
                else:
                    sv_list.append([sv[sv_id[k]], ' '.join(sv[sv_id[k] + 1:sv_id[k + 1]])])
                k += 1
            if act in da_dict.keys():
                da_dict[act] += sv_list
            else:
                da_dict[act] = sv_list
        return da_dict

    def decode(self, ids, skip_special_tokens=False):

        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        def list_duplicates_of(seq, item):
            start_at = -1
            locs = []
            while True:
                try:
                    loc = seq.index(item, start_at + 1)
                except ValueError:
                    break
                else:
                    locs.append(loc)
                    start_at = loc
            return locs

        for st in self.SPECIAL_TOKENS:
            indices = list_duplicates_of(text, st)
            if indices:
                indices.sort()
                index_count = 0
                for index in indices:
                    real_index = index + index_count
                    text = text[:real_index] + ' ' + text[real_index:]
                    text = text[:real_index + len(st) + 1] + ' ' + text[real_index + len(st) + 1:]
                    index_count += 2
        text = text.replace('  ', ' ')
        return text

    def convert_act(self, dialog_act):

        bs = []
        for d in dialog_act:
            dom, act = d.split('-')
            bs.append(dom.lower())
            bs.append(act.lower())
            for slot, value in dialog_act[d]:
                bs.append(slot.lower())
                if isinstance(value, dict):
                    for k in value.keys():
                        bs.append(k)
                        bs.append(value[k])
                else:
                    bs.append(value.lower())

        return bs

    def convert_value(self, da_dict, constraints, kb, whole_kb):
        '''KB modifies the raw dialog act, e.g., by replacing it with nooffer when the kb is empty.
        In addition, the placeholder slot values are replaced with real ones as determined by the kb.'''
        if kb is None:
            tmp = {}
            tmp['{}-nooffer'.format(self.cur_dom)] = constraints
            da_dict = tmp
        else:
            del_key = []
            for dom_act in da_dict.keys():
                # eliminate dialog acts that are empty
                if dom_act == '':
                    del_key.append(dom_act)
                    continue
                # eliminate nooffer and nobook when db is not empty
                if dom_act.split('-')[1] in ['nobook', 'nooffer']:
                    del_key.append(dom_act)
                    continue
                for i, sv in enumerate(da_dict[dom_act]):
                    key = sv[0]

                    if 'hotel' in dom_act and key == 'price':
                        key = 'pricerange'
                    # fix the WRONGLY generated slot values
                    if key in kb.keys():
                        if da_dict[dom_act][i][1] != '?':
                            if not key in ['ref', 'phone', 'id', 'post', 'addr', 'name']:
                                da_dict[dom_act][i][1] = kb[key]
                    # canonicalize the slot values
                    elif key == 'area':
                        for area in ["centre", "east", "south", "west", "north"]:
                            if area in sv[1]:
                                da_dict[dom_act][i][1] = area
                    elif key == 'price':
                        for price in ["cheap", "expensive", "moderate", "free"]:
                            if price in sv[1]:
                                da_dict[dom_act][i][1] = price
                    elif key == 'ticket':
                        if 'gbp' in sv[1]:
                            da_dict[dom_act][i][1] = sv[1].replace('gbp', 'pounds')
                    elif key == 'choice':
                        if sv[1].isdigit():
                            da_dict[dom_act][i][1] = str(len(whole_kb))

            for key in del_key:
                if key.split('-')[0] == 'train':
                    da_dict['train-offerbook'] = [['ref', '[train_reference]']]
                elif key.split('-')[0] == 'nooffer':
                    da_dict['{}-inform'.format(self.cur_dom)] = da_dict[key]
                da_dict.pop(key, None)

        return da_dict

    def convert_kb(self, kb_results):
        # Convert the kb results to the format used by this program
        new_kb = {}
        for key in kb_results:
            value = kb_results[key]
            if key == 'arriveBy':
                key = 'arrive'
            elif key == 'leaveAt':
                key = 'leave'
            elif key == 'trainID':
                key = 'id'
            elif key == 'Ref':
                key = 'ref'
            elif key == 'address':
                key = 'addr'
            elif key == 'duration':
                key = 'time'
            elif key == 'postcode':
                key = 'post'
            new_kb[key] = value
        return new_kb

    def top_filtering(self, logits, threshold=-float('Inf'), filter_value=-float('Inf')):

        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        self.top_k = min(self.top_k, logits.size(-1))
        if self.top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value
        return logits

    def init_session(self):
        self.reset()

    def reset(self):
        self.t = 0
        self.history = []
        self.cur_dom = ''
        self.prev_dom = ''

    def predict(self, usr):
        self.t += 1
        self.history.append(self.tokenizer.encode(usr.lower()))
        # decode the first token to determine it's chitchat or TOD
        with torch.no_grad():
            out_ids, dialog_act, cs_dict, kb_results, whole_kb = self.sample_sequence(self.history, mode=self.mode)
        self.history.append(out_ids)
        out_text = self.decode(out_ids, skip_special_tokens=False)
        # self.logger.info('self.history:')
        # self.logger.info(self.history)
        self.logger.info('cs :', cs_dict)
        self.logger.info('act :', self.decode(dialog_act))
        self.logger.info('kb :', kb_results)
        self.logger.info('cur_dom:', self.cur_dom)
        out_text = self.postprocess(out_text, kb_results, whole_kb)
        self.history = self.history[-(2 * self.max_history + 1):]
        return out_text

    def infer(self, history):
        ''' Infer the output segments based on history'''
        history_ids = []
        for line in history:
            history_ids.append(self.tokenizer.encode(line.lower()))
        with torch.no_grad():
            out_ids, dialog_act, cs_dict, kb_results, whole_kb = self.sample_sequence(history_ids, mode=self.mode)
        out_text = self.decode(out_ids, skip_special_tokens=False)
        self.logger.info('cs :', cs_dict)
        self.logger.info('act :' + self.decode(dialog_act))
        self.logger.info('kb :'+ str(kb_results))
        self.logger.info('cur_dom:' + self.cur_dom)
        out_text = self.postprocess(out_text, kb_results, whole_kb)
        return out_text

    def infer_fused(self, history):
        ''' Infer the output segments based on history'''
        history_ids = []
        for line in history:
            history_ids.append(self.tokenizer.encode(line.lower()))
        with torch.no_grad():
            out_ids, dialog_act, cs_dict, kb_results, whole_kb, conversation_mode = self.sample_sequence(history_ids, mode=self.mode)
        out_text = self.decode(out_ids, skip_special_tokens=False)
        self.logger.info('cs :', cs_dict)
        self.logger.info('act :', self.decode(dialog_act))
        self.logger.info('kb :', kb_results)
        self.logger.info('cur_dom:', self.cur_dom)
        out_text = self.postprocess(out_text, kb_results, whole_kb)
        return out_text, conversation_mode
    
    def infer_cs(self, history_string_version):
        ''' Infer the output segments based on history '''
        if self.mode == 'chitchat_single':
            sys.exit("No dialogue states for chitchat-only ")
        history_ids = []
        for turn in history_string_version:
            history_ids.append(self.tokenizer.encode(turn.lower()))
        with torch.no_grad():
            _, _, cs_dict, _, _ = self.sample_sequence(history_ids, mode=self.mode)
        return cs_dict

    def infer_cs_and_response(self, history_string_version):
        ''' Infer the output segments based on history '''
        if self.mode == 'chitchat_single':
            sys.exit("No dialogue states for chitchat-only ")
        history_ids = []
        self.logger.info('history_string_version:')
        self.logger.info(history_string_version)
        for turn in history_string_version:
            history_ids.append(self.tokenizer.encode(turn.lower()))
        with torch.no_grad():
            if self.mode == 'fused':
                out_ids, _, cs_dict, _, _, _ = self.sample_sequence(history_ids, mode=self.mode)
            else:
                out_ids, _, cs_dict, _, _ = self.sample_sequence(history_ids, mode=self.mode)
        out_text = self.decode(out_ids, skip_special_tokens=False)
        return cs_dict, out_text

    def postprocess(self, out_text, kb_results, whole_kb):
        ''' Postprocess the output text by replacing the entities in the out_text with the KB results 
        '''
        self.logger.info('kb_results in postprocess:')
        self.logger.info(kb_results)
        self.logger.info('whole_kb in postprocess:')
        self.logger.info(whole_kb)
        self.logger.info('out_text before postprocess:')
        self.logger.info(out_text)
        # heuristics
        if 'center of town' in out_text:
            out_text = out_text.replace('center of town', 'centre')
        if 'south part of town' in out_text:
            out_text = out_text.replace('south part of town', 'south')
        if 'no entrance fee' in out_text:
            out_text = out_text.replace('no entrance fee', 'free')
        if 'free to enter' in out_text:
            out_text = out_text.replace('free to enter', 'free')
        if 'No entrance fee' in out_text:
            out_text = out_text.replace('No entrance fee', 'free')
        sv = ['reference', 'id', 'postcode', 'phone', 'addr', 'name']
        slots = ['[' + self.cur_dom + '_' + s + ']' for s in sv]
        default_value = {'ref': '00000000', 'id': 'tr7075', 'post': 'cb21ab', 
                         'phone': '01223351880', 'name': 'error',
                         'addr': "Hills Rd , Cambridge"}
        for slot, s in zip(slots, sv):
            if s == 'reference':
                t = 'ref'
            elif s == 'postcode':
                t = 'post'
            else:
                t = s
            if out_text.count(slot) > 1:
                self.logger.info('more than one of this slot:')
                self.logger.info(slot)
                try:
                    if len(kb_results) >= 1:
                        self.logger.info('more than one entries in kb_result')
                        out_tok = []
                        tmp = copy.deepcopy(out_text).split(' ')
                        k = 0
                        for tok in tmp:
                            if tok == slot:
                                out_tok.append(self.convert_kb(whole_kb[k])[t])
                                k += 1
                            else:
                                out_tok.append(tok)
                            out_text = ' '.join(out_tok)
                    else:
                        self.logger.info('no entries in kb_result')
                        out_text = out_text.replace(slot, default_value[t])
                except:
                    # when whole_kb is exhausted, 
                    # force ignore the rest of the out_text
                    out_text = out_text.replace(slot, default_value[t])
            else:
                try:
                    if slot == '[taxi_phone]':
                        out_text = out_text.replace(slot, ''.join(kb_results['taxi_phone']))
                    else:
                        out_text = out_text.replace(slot, kb_results[t])
                except:
                    self.logger.info('default value is used for this slot:')
                    self.logger.info(slot)
                    out_text = out_text.replace(slot, default_value[t])

            
        return out_text.strip()


