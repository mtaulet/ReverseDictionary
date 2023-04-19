from transformers import BertTokenizer, BertModel
from torch.nn import functional as F
import torch
import torch.nn as nn


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.bert_backend = BertModel.from_pretrained("bert-base-uncased")
        self.lstm_decoder = nn.LSTM(768, 100, num_layers=4, dropout=0.1)

    def forward(self, input_ids, attn_mask):
        bert_out = self.bert_backend(input_ids, attention_mask=attn_mask)
        last_hidden = bert_out[0]
        # By default LSTM expects an input of (seq_len, batch_size, hidden_dim)
        last_hidden = last_hidden.transpose(0, 1)
        out, _ = self.lstm_decoder(last_hidden)
        # Output is of (seq_len, batch_size, H_out). We only want the final token in the sequence.
        out = out[-1]  # (batch_size, H_out)
        return out

