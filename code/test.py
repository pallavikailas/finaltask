import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from tqdm import tqdm
from torch import nn

device='cpu'

model=BERTClassWithFusion()
model.load_state_dict(torch.load('/kaggle/working/NLP_model.pt'))

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
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)  # Transfer targets to GPU
            outputs = model(ids, mask, token_type_ids)
            all_targets.append(targets)
            all_outputs.append(torch.sigmoid(outputs))
            
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()  # Transfer back to CPU for metrics
        all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()  # Transfer back to CPU for metrics
            
        binary_outputs = (all_outputs >= 0.5).astype(int)
        accuracy = metrics.accuracy_score(all_targets, binary_outputs)
        f1_score_macro = metrics.f1_score(all_targets, binary_outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Macro) = {f1_score_macro}")

testing(model)