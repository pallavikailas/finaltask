import numpy as np
import pandas as pd
import transformers
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

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