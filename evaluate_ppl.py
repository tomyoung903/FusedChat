'''
    Evaluate perplexity on the test set. The model can be either fused or classification-based.
    In the case of fused, use_classifier is set to False. In the case of classification-based, use_classifier is set to True.
    In the case of classification-based, cls_model_checkpoint is specified. 
    And model_checkpoint (and weights_name) refers to the checkpoint of the chitchat_single model
    In the case of fused, model_checkpoint (and weights_name) refers to the checkpoint of the fused model.
    tensor_cache_for_test must follow the format set in train.py.
'''

import logging
from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
from pytorch_transformers import GPT2DoubleHeadsModel
import math
from pytorch_transformers import GPT2Tokenizer
from util import SPECIAL_TOKENS_plus_chitchat_sor, SPECIAL_TOKENS_chitchat_single
from conversation_mode_classification import ModeClassification

def evaluate_ppl():
    parser = ArgumentParser()
    parser.add_argument("--type_of_system", type=str, default='classification-based',
                        help="the type of system is either classification-based or fused.")
    parser.add_argument("--cls_model_checkpoint", type=str, default="cls_mdls/multi_turn_epoch_9.mdl", 
    help="Path of the classification model checkpoint")
    parser.add_argument("--model_checkpoint", type=str, default="runs/chitchat_single_nov_26", 
    help="In the case of classification-based, \
    And model_checkpoint (and weights_name) refers to the checkpoint of the chitchat_single model \
    In the case of fused, model_checkpoint (and weights_name) refers to the checkpoint of the fused model.")
    parser.add_argument("--weights_name", type=str, required=False, 
                    default='checkpoint_mymodel_8.pth', \
                    help="In the case of classification-based, \
        And model_checkpoint (and weights_name) refers to the checkpoint of the chitchat_single model \
        In the case of fused, model_checkpoint (and weights_name) refers to the checkpoint of the fused model.")
    parser.add_argument("--cls_max_length", type=int, required=False, \
                    default=256)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for test")
    parser.add_argument("--eval_out", type=str, default='outs/ppl/cls_dec_6.txt', help="")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--tensor_cache_for_test", type=str, 
                        default='./data_cache/chitchat_single_nov_26_tensor_cache_test',
                        help="tensor_cache_for_test must follow the format of fused or chitchat_single.")

    args = parser.parse_args()

    if args.type_of_system == 'classification-based':
        args.use_classifier = True
    elif args.type_of_system == 'fused':
        args.use_classifier = False

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.eval_out)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Load pretrained model
    model_class = GPT2DoubleHeadsModel
    model, _ = model_class.from_pretrained(args.model_checkpoint, weights_name=args.weights_name)
    model.to(args.device)
    if args.use_classifier:
        cls_model = ModeClassification(args.cls_model_checkpoint, args.cls_max_length, args.device)

    # Load tokenizer. In the case of classification-based, the tokenizer is the one of the chitchat_single model.
    # In the case of fused, the tokenizer is the one of the fused model.
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint, unk_token='<|unkwn|>')
    
    SPECIAL_TOKENS_DICT = {}
    if args.use_classifier:
        for st in SPECIAL_TOKENS_chitchat_single:
            SPECIAL_TOKENS_DICT[st] = st
    else:
        for st in SPECIAL_TOKENS_plus_chitchat_sor:
            SPECIAL_TOKENS_DICT[st] = st
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    # identify the <chitchat> token for the fused model
    if not args.use_classifier:
        chitchat_token_id = tokenizer.convert_tokens_to_ids('<chitchat>')
    
    # Load dataset
    tensor_dataset = torch.load(args.tensor_cache_for_test)
    
    # Set up loss function
    cel = torch.nn.CrossEntropyLoss(ignore_index=-1)
    softmax = torch.nn.Softmax(dim=1)

    def avg_nll_fused(model, batch):
        model.eval()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        with torch.no_grad():
            input_ids, lm_labels, token_type_ids = batch
            model_outputs = model(input_ids, token_type_ids=token_type_ids)
            lm_logits = model_outputs[0]
            lm_logits_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_shifted = lm_labels[..., 1:].contiguous().view(-1)
            probs = softmax(lm_logits_shifted)
            chitchat_token_id = tokenizer.convert_tokens_to_ids('<chitchat>')
            chitchat_index = (lm_labels_shifted == chitchat_token_id).nonzero()[0][0]
            # the classification loss
            prob_cc = probs[chitchat_index][chitchat_token_id]

            # calculate the negative log likelihood for tokens after <chitchat> and <system>
            all_nlls = []
            for i in range(chitchat_index+2, lm_labels_shifted.shape[0]):
                if lm_labels_shifted[i] != -1:
                    all_nlls.append(-math.log(probs[i, lm_labels_shifted[i]]))
            avg_nll = sum(all_nlls)/len(all_nlls)
            
            # adjust for classification loss
            avg_nll_ad = avg_nll - math.log(prob_cc)
            return avg_nll_ad

    def avg_nll_cls(model, batch, prob_cc):
        model.eval()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        with torch.no_grad():
            input_ids, lm_labels, token_type_ids = batch
            model_outputs = model(input_ids, token_type_ids=token_type_ids)
            lm_logits = model_outputs[0]
            lm_logits_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_shifted = lm_labels[..., 1:].contiguous().view(-1)
            avg_nll = cel(lm_logits_shifted, lm_labels_shifted)
            # adjust for classification loss
            avg_nll_ad = avg_nll - math.log(prob_cc)
            return avg_nll_ad

    def decode(input_ids):
        # decode the input_ids into lists of tokens
        token_list = []
        for token_id in input_ids:
            token_list.append(int(token_id))
        s = tokenizer.decode(token_list)
        s = s.replace('<bos>', '')
        s = s.replace('<pad>', '')
        s = s.replace('<eos>', '')
        # <sor> and <chitchat> are only used in the fused model
        s = s.replace('<sor>', '')
        s = s.replace('<chitchat>', '')
        s = s.replace('<user>', '<delimiter>')
        s = s.replace('<system>', '<delimiter>')
        s_split = s.split('<delimiter>')
        return s_split[1:]

    avg_nll_list = []
    for batch_idx in tqdm(range(tensor_dataset[0].size()[0])): # (range(len(loader))):
        if args.use_classifier:
            input_ids = tensor_dataset[0][batch_idx][1]
            # decode it
            history_and_response = decode(input_ids)
            history = history_and_response[:-1]
            logits = cls_model.infer_logits(history)
            logger.info('logits: {}'.format(logits))
            probs = softmax(logits)
            logger.info('probs: {}'.format(probs))
            if probs[0][0] > probs[0][1]:
                logger.info('the mode is chitchat')
            else:
                logger.info('the mode is tod')
        if not args.use_classifier:
            # only evaluate the batch with the <chitchat> token in the input_ids
            if chitchat_token_id not in tensor_dataset[0][batch_idx][1]:
                continue
        batch = torch.unsqueeze(tensor_dataset[0][batch_idx][1], 0), \
                torch.unsqueeze(tensor_dataset[2][batch_idx][1], 0), \
                torch.unsqueeze(tensor_dataset[4][batch_idx][1], 0)
        if args.use_classifier:
            avg_nll_list.append(avg_nll_cls(model, batch, probs[0][0]))
        else:
            avg_nll_list.append(avg_nll_fused(model, batch))

    dataset_ppl = math.exp(sum(avg_nll_list)/len(avg_nll_list))
    logger.info('dataset_ppl: {}'.format(dataset_ppl))
    # save args
    args_dict = vars(args)
    args_dict_string = {key:str(value) for key, value in args_dict.items()}
    logger.info('args_dict: {}'.format(str(args_dict_string)))
if __name__ == "__main__":
    evaluate_ppl()


