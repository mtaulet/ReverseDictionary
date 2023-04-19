from transformers import BertTokenizer, BertModel
from torch.nn import functional as F
import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.bert_backend = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(self.bert_backend.config.hidden_size, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128 * 200)
        self.dropout1 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(128 * 200, 2048)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(2048, 512)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(512, 100)

    def forward(self, input_ids, attn_mask):
        bert_out = self.bert_backend(input_ids, attention_mask=attn_mask)
        last_hidden = bert_out[0]
        out = self.relu1(self.linear1(last_hidden))
        out = self.relu2(self.linear2(out))
        out = out.view(-1, 128 * 200)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu3(self.linear3(out))
        out = self.relu4(self.linear4(out))
        out = self.linear5(out)
        return out
