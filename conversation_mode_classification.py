import torch.nn as nn
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import math
from tqdm import tqdm
from sklearn import metrics

from bert_input_utils import *


class BertEncoder(nn.Module):
    def __init__(self, embd_dim, num_classes, device='cuda'):
        super(BertEncoder, self).__init__()
        self.device = device
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.logfilename = 'bert_cross_encoder.txt'
        self.log_file = open(self.logfilename, 'w')
        '''
            for p in self.bert_encoder.parameters():
                p.requires_grad = False
        '''
        self.fc_o = nn.Linear(embd_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        # print(self.parameters())
        self.optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=0.000005)

    def _to_tensor(self, x):
        if self.device == 'cuda':
            return torch.tensor(x).cuda()
        else:
            return torch.tensor(x)

    def encode(self, xs, dialog_sent_masks):
        # xs: batch_size x max_sent_length
        # dialog_sent_masks : batch_size x max_sent_length
        # self.eval()
        # batch_size, sent_len = xs.size()
        xs = torch.unsqueeze(xs, 0)
        dialog_sent_masks = torch.unsqueeze(dialog_sent_masks, 0)
        encoded_layers, _ = self.bert_encoder(xs, attention_mask=dialog_sent_masks, output_all_encoded_layers=False)
        # encoded_layers: batch_size x max_sent_length x 768
        return encoded_layers

    def train_model(self, cs, ys, mask_c, batch_size=16):
        self.train()
        num_batches = math.ceil(float(len(ys)) / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(cs))
            if end - start != batch_size:
                continue
            cs_ = self._to_tensor(cs[start:end]).long()
            ys_ = self._to_tensor(ys[start:end]).long()
            mask_c_ = self._to_tensor(mask_c[start:end])
            cs_vecs = self.encode(cs_, mask_c_)

            logits = self.fc_o(cs_vecs[:, 0])  # batch_size * num_classes
            loss = nn.CrossEntropyLoss()
            output = self.loss(logits, ys_)
            output.backward()
            self.log_file.write(str(output) + '\n')
            self.log_file.write('\n\n')
            torch.nn.utils.clip_grad_norm(self.parameters(), 0.01)
            self.optimizer.step()
            self.zero_grad()

    def test(self, cs, ys, mask_c, batch_size=4):
        self.eval()
        num_batches = math.ceil(float(len(cs)) / batch_size)
        gts = []
        lgts = []
        predictions = []
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(cs))
            if end - start != batch_size:
                continue
            cs_ = self._to_tensor(cs[start:end]).long()
            ys_ = self._to_tensor(ys[start:end]).long()
            # print(ys_)
            mask_c_ = self._to_tensor(mask_c[start:end])
            cs_vecs = self.encode(cs_, mask_c_)
            logits = self.fc_o(cs_vecs[:, 0])  # batch_size * num_classes
            # print(logits)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(list(preds.detach().cpu().numpy()))
            gts.extend(list(ys_.detach().cpu().numpy()))
            lgts.extend(list(logits.detach().cpu().numpy()))
            # gts.extend(list(ys_))
            # print(preds)
        # print(gts)
        # print(predictions)
        acc = metrics.accuracy_score(gts, predictions)
        report = metrics.classification_report(gts, predictions, digits=4)
        return acc, report, gts, predictions

    def infer(self, cs, mask_c):
        self.eval()
        cs_ = self._to_tensor(cs).long()
        mask_c_ = self._to_tensor(mask_c)
        cs_vecs = self.encode(cs_, mask_c_)
        logits = self.fc_o(cs_vecs[:, 0])  # batch_size * num_classes
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def infer_logits(self, cs, mask_c):
        self.eval()
        cs_ = self._to_tensor(cs).long()
        mask_c_ = self._to_tensor(mask_c)
        cs_vecs = self.encode(cs_, mask_c_)
        logits = self.fc_o(cs_vecs[:, 0])  # batch_size * num_classes
        return logits


class ModeClassification:
    def __init__(self, model_checkpoint, max_length, device='cuda'):
        self.model = BertEncoder(768, 2, device)
        if device == 'cuda':
            self.model = self.model.cuda()
        device = torch.device(device)
        self.model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def classify(self, history):
        if type(history) != list:
            exit('history has to be a list!')
        ids, masks = get_ids_and_masks_backward(history, self.max_length, self.tokenizer)
        # print('ids:')
        # print(ids)
        # print('masks:')
        # print(masks)
        label = self.model.infer(ids, masks)
        return label

    def infer_logits(self, history):
        if type(history) != list:
            exit('history has to be a list!')
        ids, masks = get_ids_and_masks_backward(history, self.max_length, self.tokenizer)
        # print('ids:')
        # print(ids)
        # print('masks:')
        # print(masks)
        logits = self.model.infer_logits(ids, masks)
        return logits
