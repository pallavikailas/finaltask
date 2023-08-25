import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["text"], item["is_humor"]

# Load the CSV file
train = pd.read_csv(r'C:\Users\KRISH DIDWANIA\final_task-3\preprocessed_train.csv')

# Select required columns and convert to a list of dictionaries
train_data = train[["text", "is_humor"]].to_dict(orient='records')
# Create an instance of your CustomDataset
custom_dataset = CustomDataset(train_data)

# Define DataLoader parameters
batch_size = 16
shuffle = True  # Set to True for training, False for validation/testing

# Create a DataLoader instance
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    
           