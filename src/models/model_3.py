from transformers import BertTokenizer, BertModel
from torch.nn import functional as F
import torch
import torch.nn as nn


class Model3(nn.Module):
    """Variation:
    - Use Tanh instead of ReLU, since word embeddings should allow negative values. Tanh is a shifted sigmoid to [-1,1],
      so it should support negative values
    - Increase dropout prob to 0.5
    - Change BatchNorm to LayerNorm
    - Use a Multi-Attention Head and add attention to the dimensionality-reduction before the LayerNorm
    """
    def __init__(self):
        super(Model3, self).__init__()
        self.bert_backend = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(self.bert_backend.config.hidden_size, 256)
        self.tanh1 = nn.Tanh()
        self.mha1 = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.ln1 = nn.LayerNorm((200, 256))
        self.linear2 = nn.Linear(256, 128)
        self.tanh2 = nn.Tanh()
        self.mha2 = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
        self.ln2 = nn.LayerNorm((200, 128))
        self.dropout1 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128 * 200, 2048)
        self.tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(2048, 512)
        self.tanh4 = nn.Tanh()
        self.linear5 = nn.Linear(512, 100)

    def forward(self, input_ids, attn_mask):
        bert_out = self.bert_backend(input_ids, attention_mask=attn_mask)
        last_hidden = bert_out[0]

        # Dimensionality reduction with attention
        out = self.tanh1(self.linear1(last_hidden))
        attn_out, attn_weights = self.mha1(out, out, out)
        out = out + attn_out
        out = self.ln1(out)

        out = self.tanh2(self.linear2(out))
        attn_out, attn_weights = self.mha2(out, out, out)
        out = out + attn_out
        out = self.ln2(out)

        out = out.view(-1, 128 * 200)
        out = self.dropout1(out)

        # Convert to embedding
        out = self.tanh3(self.linear3(out))
        out = self.tanh4(self.linear4(out))
        out = self.linear5(out)
        return out
