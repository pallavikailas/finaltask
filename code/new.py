# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv(r'/kaggle/input/dataset/preprocessed_train1.csv')

new_df = df[['text', 'is_humor']].copy()

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 4
EPOCHS = 10
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
            return_token_type_ids=True)
        
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


class BERTClassWithFusion(nn.Module):
    def __init__(self):
        super(BERTClassWithFusion, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.l3_bilstm = nn.Linear(256, 6)
        self.l3_linear = nn.Linear(768, 6)  # Linear layer for BERT output
        self.l4 = nn.Linear(12, 1)
        
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        last_hidden_state = output_1.last_hidden_state
        
        # Linear transformation of BERT output
        bert_output = self.l3_linear(last_hidden_state)
        
        output_2 = self.l2(last_hidden_state)
        lstm_output, _ = self.lstm(output_2)  # Use LSTM layer
        
        # Linear transformation of BiLSTM output
        lstm_output = self.l3_bilstm(lstm_output)
        
        # Concatenate BERT output and BiLSTM output
        fused_output = torch.cat((bert_output, lstm_output), dim=2)
    
        # Reshape the fused output to match the batch size
        fused_output = fused_output.view(ids.size(0), -1, 12)
    
        # Sum along the sequence length dimension
        fused_output = torch.sum(fused_output, dim=1)

        fused_output = self.l4(fused_output)
        return fused_output

# Create an instance of the model
model = BERTClassWithFusion()

model.to(device)

def loss_fn(outputs, targets):
    outputs = outputs.view(-1, 1)
    targets = targets.view(-1, 1)
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

df1 = pd.read_csv(r'/kaggle/input/devset/dev.csv')

new_df1 = df1[['text', 'is_humor']].copy()

dev_dataset = new_df1.sample(frac=1, random_state=200) 
dev_set = CustomDataset(dev_dataset, tokenizer, MAX_LEN)  
print("FULL Dataset: {}".format(new_df1.shape)) 

dev_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

dev_loader = DataLoader(dev_set, **dev_params)


def validating(model):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(dev_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        loss=loss_fn(outputs,targets)
        optimizer.zero_grad()
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out, targ = validating(model)
    out= (np.array(out) >= 0.5).astype(int)
    accuracy = metrics.accuracy_score(targ, out)
    f1_score_micro = metrics.f1_score(targ, out, average='micro')
    f1_score_macro = metrics.f1_score(targ, out, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

for epoch in tqdm(range(EPOCHS)):
    train(epoch) 

