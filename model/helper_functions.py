from torch.utils.data import Dataset

class Stage1ExampleDataset(Dataset):
    """
    Fixed stage-1 dataset built from precomputed clean example metadata.

    Each item is a dict with:
        songId, centerChunk, memberName, label, weight, source
    """
    def __init__(self, examples):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]