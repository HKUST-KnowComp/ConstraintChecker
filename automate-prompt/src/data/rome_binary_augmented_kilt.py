import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
        all_views=False,
        result_alt=False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.result_alt = result_alt

        with jsonlines.open(data_path) as f:
            for d in f:
                if len(d["alternatives"]) > 0 and len(d["filtered_rephrases"]) > 0:
                    self.data.append(
                        {
                            **{
                                k: d[k]
                                for k in (
                                    "input",
                                    "prediction",
                                    "alternatives",
                                    "new_id",
                                )
                            },
                            "label": d["output"][0]["answer"],
                        }
                    )

        self.max_length = max_length
        self.all_views = all_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        output = {
            "src": self.data[item]["input"],
            "pred": self.data[item]["prediction"] == "SUPPORTS",
            "alt": self.data[item]["alternatives"][0] == "SUPPORTS",
            "id": self.data[item]["new_id"]
        }

        return output

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        labels = [b['alt'] if self.result_alt else b['pred'] for b in batch]
        
        batches = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": src,
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["labels"] = torch.tensor(labels).float()
        batches["raw"] = batch

        return batches
