from datasets import Dataset


class CombinedDataset(Dataset):
    def __init__(
        self, 
        dataset_1: Dataset, 
        dataset_2: Dataset,
    ):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2

    def __len__(self):
        return min(len(self.dataset_1), len(self.dataset_2))

    def __getitem__(self, item):
        return self.dataset_1[item], self.dataset_2[item]

    def collate_fn(self, batch, proportion_2=0.06):
        from_1 = [x[0] for x in batch]
        from_2 = [x[1] for x in batch]
        # from_2 = from_2[:int(len(from_2) * proportion_2)]
        return self.dataset_1.collate_fn(from_1), self.dataset_2.collate_fn(from_2)
