import torch
import torch.optim as optim
import torch.nn as nn
from model import BERT  # Assuming you have a model.py file with your BERT model implementation
import pandas as pd
from torch.utils.data import DataLoader
from dataloader import CustomDataset

# Define your hyperparameters
vocab_size = 10000
d_model = 768 # Set d_model to 768 for the final hidden vector
n_layers = 12
n_heads = 12
max_len = 8000
batch_size = 1
actual_batch_size=1
learning_rate = 0.001
num_epochs = 3
d_ff = 2048  # Define the value of d_ff
dropout = 0.1  # Define the value of dropout

# Create the BERT model
bert_model = BERT(vocab_size, d_model, n_layers, n_heads, max_len, d_ff, dropout).to('cuda')

# Define your loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=learning_rate)


# Load your training data
train = pd.read_csv(r'C:\Users\KRISH DIDWANIA\final_task-3\preprocessed_train.csv')

# Select required columns and convert to a list of dictionaries
train_data = train[["text", "is_humor"]].to_dict(orient='records')

# Preprocess the train_data to match the expected format
preprocessed_train_data = []
for item in train_data:
    preprocessed_item = {
        "text": item["text"],    # Replace with your text preprocessing steps if needed
        "is_humor": item["is_humor"]
    }
    preprocessed_train_data.append(preprocessed_item)

# Create an instance of your CustomDataset using the preprocessed data
custom_dataset = CustomDataset(preprocessed_train_data)
print(custom_dataset[3])
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0

    for batch_texts, batch_labels in train_loader:

        batch_texts=batch_texts.to('cuda')
        batch_labels=batch_labels.to('cuda')

        optimizer.zero_grad()
        input_ids = torch.randint(0, vocab_size, (actual_batch_size, max_len))
        segment_ids = torch.randint(0, 2, (actual_batch_size, max_len))
    # Tokenize and preprocess the texts (Replace this with actual tokenization)
     
        # Convert labels to tensor
        # Convert labels to tensor
        batch_labels.clone().detach()
        label_tensor=torch.tensor(batch_labels,dtype=torch.float32).view(-1, 1)
    # Forward pass
        output = bert_model(input_ids, segment_ids)

    # Reshape output and labels
        output = output.view(-1, d_model)   # Reshape to (batch_size * max_len, dd_model)
        label_tensor = label_tensor.repeat(1,max_len) # Repeat labels for each position
    # Calculate the loss
    

# Define a linear layer for classification
        classification_layer = nn.Linear(768,1)

# Apply the linear layer to the input tensor
        output_tensor = classification_layer(output)
      
# You can also apply softmax if needed for classification probabilities
        loss = criterion(output_tensor, label_tensor)

    # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")



print(output.shape)  # This will print the shape of the final output tensor