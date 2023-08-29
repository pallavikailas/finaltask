import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from tqdm import tqdm
from torch import nn
from model import BERTClassWithFusion
from dataloader import CustomDataset

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model=BERTClassWithFusion()
model.load_state_dict(torch.load('/kaggle/input/trainedmodel/NLP_model.pt'))
model.to(device)

df2 = pd.read_csv(r'/kaggle/input/nlpfinal/preprocessed_test.csv')

new_df2 = df2[['text', 'is_humor']].copy()

test_dataset = new_df2.sample(frac=1, random_state=200) 
test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)  
print("FULL Dataset: {}".format(new_df2.shape)) 

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_loader = DataLoader(test_set, **test_params)

def testing(model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)  # Transfer targets to GPU
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

out2, targ2 = testing(model)
out2= (np.array(out2) >= 0.5).astype(int)    
accuracy2 = metrics.accuracy_score(targ2, out2)    
f1_score_macro2 = metrics.f1_score(targ2, out2, average='macro')
print(f"Accuracy Score Test = {accuracy2}")
print(f"F1 Score (Macro) Test = {f1_score_macro2}")