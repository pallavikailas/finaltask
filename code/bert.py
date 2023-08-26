# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv(r'/Users/admin/Documents/Manipal/MRM/FINAL_TASK/final_task/preprocessed_train.csv')

new_df = df[['text', 'is_humor']].copy()

MAX_LEN = 200
TRAIN_BATCH_SIZE = 1600
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets = self.data.is_humor
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):

        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

train_dataset=new_df.sample(random_state=200) 
train_dataset = new_df.sample(frac=1, random_state=200) 
training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)  
print("FULL Dataset: {}".format(new_df.shape)) 

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.l3 = torch.nn.Linear(256, 6)  # Update the input size to 256 due to bidirectional LSTM
        self.l4 = torch.nn.Linear(6, 1)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        last_hidden_state = output_1.last_hidden_state
        output_2 = self.l2(last_hidden_state)
        lstm_output, _ = self.lstm(output_2)  # Use LSTM layer
        output = self.l3(lstm_output)
        output = self.l4(output)
        return output

model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    outputs = outputs.view(-1, 1)
    targets = targets.view(-1, 1)
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)    
