import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd 

class MyDataset(Dataset):
    def __init__(self, root_dir, file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(file)
        self.transform = transform

        self.text = self.df["text"]
        self.y = self.df.drop(["text"], axis=1)

    def __len__():
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        y = self.y[index]

        if self.transform is not None:
            text = self.transform(text)

        numericalized_text = [self.vocab.stoi["<SOS>"]]
        numericalized_text += self.vocab.numericalize(text)
        numericalized_text.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_text), y
