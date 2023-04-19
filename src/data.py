import os

import pandas as pd
import gensim.downloader as api
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset to allow for variable output types. This lets us experiment with changing what type we expect the
    outputs to be (word embeddings, BERT tokenized, etc.).
    """
    def __init__(self, input_ids, attention_masks, outputs):
        assert len(input_ids) == len(attention_masks)
        assert len(input_ids) == len(outputs)
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.outputs = outputs


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.outputs[idx]


_WORD_VECS = None
def get_wordvecs():
    global _WORD_VECS
    if _WORD_VECS is None:
        _WORD_VECS = api.load("glove-wiki-gigaword-100")
    return _WORD_VECS


def dataset_as_dataframe():
    opted_data = pd.read_pickle("datasets/preprocessed/opted.pkl")
    wordnet_data = pd.read_pickle("datasets/preprocessed/wordnet.pkl")
    return pd.concat([opted_data, wordnet_data])


def dataset_tokenized(use_cached=True):
    combined_df = dataset_as_dataframe()
    if use_cached and os.path.exists("datasets/preprocessed/input_ids_encoded.tensor") and \
            os.path.exists("datasets/preprocessed/attention_masks.tensor") and \
            os.path.exists("datasets/preprocessed/labels.tensor"):
        print("Detected cached dataset...")
        input_ids = torch.load("datasets/preprocessed/input_ids_encoded.tensor")
        attention_masks = torch.load("datasets/preprocessed/attention_masks.tensor")
        labels = torch.load("datasets/preprocessed/labels.tensor")
        print("Loaded dataset cache.")
    else:
        # Convert outputs to word vectors based on GloVe
        wordvecs = get_wordvecs()

        # Remove words that do not have GloVe embeddings
        _keep = []
        for _, word in combined_df["word"].items():  # TODO: For some reason df.series.isin doesn't work with KeyedVector
            _keep.append(word in wordvecs)
        original_size = combined_df.shape[0]
        combined_df = combined_df[_keep]
        diff = original_size - combined_df.shape[0]
        print(f"Pruned {diff} words from dataset since they do not exist in GloVe. {combined_df.shape[0]} remaining.")

        # Get output vectors
        labels = []
        for word in combined_df["word"]:
            labels.append(wordvecs[word])
        labels = torch.tensor(np.array(labels))

        input_ids = []
        attention_masks = []
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        for sent in combined_df['definition']:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(sent,                          # Sentence to encode.
                                                 add_special_tokens = True,     # Add '[CLS]' and '[SEP]'
                                                 max_length = 200,              # Pad & truncate all sentences. Max length is 291 but most are not that long
                                                 pad_to_max_length = True,
                                                 return_attention_mask = True,  # Construct attn. masks.
                                                 return_tensors = 'pt')         # Return pytorch tensors.

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        # Back up for cache
        torch.save(input_ids, "datasets/preprocessed/input_ids_encoded.tensor")
        torch.save(attention_masks, "datasets/preprocessed/attention_masks.tensor")
        torch.save(labels, "datasets/preprocessed/labels.tensor")
    return input_ids, attention_masks, labels
