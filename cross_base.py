'''Bert-based cross encoder used for sentence pair classification'''

import torch.nn as nn
import numpy as np
import torch
from pytorch_pretrained_bert import BertModel
import math
from tqdm import tqdm
from sklearn import metrics


class BertEncoder(nn.Module):
    def __init__(self, embd_dim, num_classes, option, learning_rate=5e-6):
        super(BertEncoder, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(option)
        self.fc_o = nn.Linear(embd_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=learning_rate)

    def encode(self, xs, dialog_sent_masks):
        encoded_layers, _ = self.bert_encoder(xs, attention_mask=dialog_sent_masks, output_all_encoded_layers=False)
        return encoded_layers

    def train_model(self, cs, ys, mask_c, batch_size=16):
        self.train()
        num_batches = math.ceil(float(len(ys)) / batch_size)

        def _to_tensor(x):
            return torch.from_numpy(x).cuda()
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(cs))
            if end - start != batch_size:
                continue
            cs_ = _to_tensor(cs[start:end]).long()
            ys_ = _to_tensor(ys[start:end]).long()
            mask_c_ = _to_tensor(mask_c[start:end])
            cs_vecs = self.encode(cs_, mask_c_)

            logits = self.fc_o(cs_vecs[:, 0])  # batch_size * num_classes
            loss = nn.CrossEntropyLoss()
            output = self.loss(logits, ys_)
            output.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 0.01)
            self.optimizer.step()
            self.zero_grad()

    def test(self, cs, ys, mask_c, batch_size=4):
        self.eval()
        num_batches = math.ceil(float(len(cs)) / batch_size)
        gts = []
        lgts = []
        predictions = []

        def _to_tensor(x):
            return torch.from_numpy(x).cuda()

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(cs))
            if end - start != batch_size:
                continue
            cs_ = _to_tensor(cs[start:end]).long()
            ys_ = _to_tensor(ys[start:end]).long()
            mask_c_ = _to_tensor(mask_c[start:end])
            cs_vecs = self.encode(cs_, mask_c_)
            logits = self.fc_o(cs_vecs[:, 0])
            preds = torch.argmax(logits, dim=1)
            predictions.extend(list(preds.detach().cpu().numpy()))
            gts.extend(list(ys_.detach().cpu().numpy()))
            lgts.extend(list(logits.detach().cpu().numpy()))
        acc = metrics.accuracy_score(gts, predictions)
        report = metrics.classification_report(gts, predictions, digits=4)
        return acc, report, gts, predictions
