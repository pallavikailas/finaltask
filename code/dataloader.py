import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd 
import os 

class Vocabulary:
    def __init__(self):
        # Initialize token-to-index mapping with special tokens
        self.stoi = {"<SOS>": 0, "<EOS>": 1, "<UNK>": 2}
        # Initialize index-to-token mapping
        self.itos = {0: "<SOS>", 1: "<EOS>", 2: "<UNK>"}

class CustomDataset(Dataset):

    def __init__(self, file, transform=None,vocab = None):
        current_dir = os.getcwd()

        file_path = os.path.join(current_dir, file)

        # Load the CSV file
        self.df = pd.read_csv("D:/final_task-5/preprocessed_train.csv")
        self.transform = transform

        self.vocab = vocab

        self.text = self.df["text"]
        self.y = self.df.drop(["text"], axis=1)
        
    def __len__(self):
        return len(self.df)

    def numericalize(self, text):
        return [self.vocab.stoi.get(token, self.vocab.stoi["<UNK>"]) for token in text.split()]

    def __getitem__(self, index):
        text = self.text[index]
        y = self.y.iloc[index] 

        if self.transform is not None:
            text = self.transform(text)

        numericalized_text = [self.vocab.stoi["<SOS>"]]
        numericalized_text += self.numericalize(text)
        numericalized_text.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_text), y

vocab = Vocabulary()

# Create an instance of the CustomDataset
dataset = CustomDataset(file='final_task-5/preprocessed_train.csv',vocab = vocab )

# Access an item from the dataset
sample_input, sample_label = dataset[0]

# Print the tensor
print("Sample Input Tensor:")
print(sample_input)

print("\nSample Label:")
print(sample_label)

