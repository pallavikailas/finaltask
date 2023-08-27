import numpy as np
import copy
import pandas as pd
from sklearn import metrics
import transformers
import torch
from tqdm import tqdm
from torch import nn
from dataloader import CustomDataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from model import BERTClassWithFusion

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_accuracy=0

df = pd.read_csv(r'/kaggle/input/nlpfinal/preprocessed_train.csv')

new_df = df[['text', 'is_humor']].copy()
train_dataset=new_df.sample(random_state=200) 
train_dataset = new_df.sample(frac=1, random_state=200) 
training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)  
print("FULL Dataset: {}".format(new_df.shape)) 

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)

# Create an instance of the model
model = BERTClassWithFusion()

model.to(device)

def loss_fn(outputs, targets):
    outputs = outputs.view(-1, 1)
    targets = targets.view(-1, 1)
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

df1 = pd.read_csv(r'/kaggle/input/nlpfinal/preprocessed_dev.csv')

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

best_model=BERTClassWithFusion()
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
    if(accuracy>max_accuracy):
        best_model=copy.deepcopy(cnn)
        max_accuracy=accuracy
    f1_score_macro = metrics.f1_score(targ, out, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

for epoch in tqdm(range(EPOCHS)):
    train(epoch) 

torch.save(best_model.state_dict(), './NLP_model.pt')
