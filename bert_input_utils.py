# input formatting utilities for BERT
import numpy as np


def pad_to_max_length(context_ids, max_length):
    # pad to the max_length
    padded_context_ids = np.zeros([max_length])
    for j in range(min(max_length, len(context_ids))):
        padded_context_ids[j] = context_ids[j]
    return padded_context_ids


def get_mask(context_ids, max_length):
    # pad to the max_length
    mask = np.zeros([max_length])
    for j in range(min(max_length, len(context_ids))):
        mask[j] = 1
    return mask


def get_ids_and_masks(history, max_length, tokenizer):
    # concatenate all sentences in history, separated by '[SEP]'. Overflowing max_length is possible.
    sent = ['[CLS]']
    for sentence in history:
        tokens = tokenizer.tokenize(sentence)
        sent = sent + tokens + ['[SEP]']
    ctx_ids = tokenizer.convert_tokens_to_ids(sent)
    ids = pad_to_max_length(ctx_ids, max_length)
    masks = get_mask(ctx_ids, max_length)
    return ids, masks


def get_ids_and_masks_backward(history, max_length, tokenizer):
    # concatenate sentences in history from the end one by one,
    # separated by '[SEP]'.
    sent = ['[CLS]']
    for i in range(len(history) - 1, -1, -1):
        sentence = history[i]
        tokens = tokenizer.tokenize(sentence)
        if len(sent + tokens + ['[SEP]']) > max_length:
            break
        else:
            sent = [sent[0]] + tokens + ['[SEP]'] + sent[1:]
    ctx_ids = tokenizer.convert_tokens_to_ids(sent)
    ids = pad_to_max_length(ctx_ids, max_length)
    masks = get_mask(ctx_ids, max_length)
    return ids, masks
