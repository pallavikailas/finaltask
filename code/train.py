import torch
import torch.optim as optim
import torch.nn as nn
from model import BERT

# Define your hyperparameters
vocab_size = 10000
d_model = 768  # Set d_model to 768 for the final hidden vector
n_layers = 6
n_heads = 8
max_len = 512
batch_size = 16
learning_rate = 0.001
num_epochs = 1
d_ff = 2048  # Adjust the value based on your preference
dropout = 0.1  # Adjust the value based on your preference

# Create the BERT model
bert_model = BERT(vocab_size, d_model, n_layers, n_heads, max_len)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=learning_rate)

# Example training data (input text and binary labels)
train_data = [
    ("This is a positive sentence.", 1),
    ("Negative sentiment in this text.", 0),
    ("Another positive example.", 1),
    # ... Add more training examples
]

# Training loop
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0

    for text, label in train_data:
        optimizer.zero_grad()

        # Tokenize and preprocess the text
        input_ids = torch.randint(0, vocab_size, (batch_size, max_len))
        segment_ids = torch.randint(0, 2, (batch_size, max_len))

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Forward pass
        output = bert_model(input_ids, segment_ids)

        # Calculate the loss
        loss = criterion(output, label_tensor)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")