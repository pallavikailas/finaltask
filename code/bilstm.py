import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import BertTokenizer  # You may need to install the transformers library
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

seed_num = 233
pad = "<pad>"
unk = "<unk>"
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

seed_num = 233

# Define your training data (replace this with your actual data)
train_data = [
    ("This is a positive sentence.", 1),
    ("Negative sentiment in this text.", 0),
    # ... Add more training examples
]

# Define a class to hold your configuration
class Args:
    def __init__(self):
        self.lstm_hidden_dim = random.randint(50, 200)
        self.lstm_num_layers = random.randint(1, 4)
        self.embed_num = random.randint(1000, 5000)
        self.embed_dim = random.randint(100, 200)
        self.class_num = random.randint(2, 10)
        self.dropout = random.uniform(0.1, 0.5)  # Adjust the range as needed
        self.paddingId = 0  # Set this to a specific value if needed
        self.word_Embedding = True
        # Initialize pretrained_weight with random data (you may load actual pretrained embeddings here)
        self.pretrained_weight = torch.randn(self.embed_num, self.embed_dim)

# Create an instance of Args to hold your configuration
args = Args()

# Your BiLSTM model definition here
class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=1, dropout=args.dropout, bidirectional=True, bias=False)
        print(self.bilstm)
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = y
        return logit

# Training loop
min_valid_loss = np.inf
# Initialize the model
model = BiLSTM(args)
if torch.cuda.is_available():
    model = model.cuda()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):  # Change the number of epochs as needed
    train_loss = 0.0
    for text, labels in train_data:
        if torch.cuda.is_available():
            labels = labels.cuda()
        # Tokenize the text and convert it into a tensor
        input_ids = tokenizer(text, truncation=True, padding=True, return_tensors="pt").input_ids
        input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids
    
        optimizer.zero_grad()
        target = model(input_ids)
        loss = criterion(target, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    valid_loss = 0.0
    for data, labels in valid_data:  # Define your validation dataset
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = model(data)
        loss = criterion(target, labels)
        valid_loss = loss.item() * data.size(0)

    print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_data)} \t\t Validation Loss: {valid_loss / len(valid_data)}')
    
    if valid_loss < min_valid_loss:
        print(f'Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

# Optionally, save the trained model
torch.save(model.state_dict(), 'bilstm_model.pth')