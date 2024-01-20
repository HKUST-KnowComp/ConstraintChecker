import typing
from torch.utils.data import Dataset
import torch
import pandas as pd
import json


class DomainDataset(Dataset):
    def __init__(
        self, data_path: str, label2id=None, *args, **kwargs
    ):
        self.sentences = []
        self.labels = []
        if label2id is None:
            self.label2id = {}
        else:
            self.label2id = label2id
        with open(data_path, "r") as f:
            for row in f:
                data = json.loads(row)
                self.sentences.append(data['text'])
                label = data['label']
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                self.labels.append(self.label2id[label])

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return {
            'text': self.sentences[item],
            'label': self.labels[item],
        }
        
    def collate_fn(self, batch):
        return {
            'text': [x['text'] for x in batch],
            'label': [x['label'] for x in batch],
        }
