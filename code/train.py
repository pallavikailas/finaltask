import torch
import torch.optim as optim
import torch.nn as nn
from model import BERT

# Define your hyperparameters
vocab_size = 10000
d_model = 768  # Set d_model to 768 for the final hidden vector
n_layers = 12
n_heads = 12
max_len = 512
batch_size = 16
learning_rate = 0.001
num_epochs = 3
d_ff = 2048  # Define the value of d_ff
dropout = 0.1  # Define the value of dropout

# Create the BERT model
bert_model = BERT(vocab_size, d_model, n_layers, n_heads, max_len, d_ff, dropout)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=learning_rate)

# Example training data (input text and binary labels)
train_data = [
    ("This is a positive sentence.", 1),
    ("Negative sentiment in this text.", 0),
    # ... Add more training examples
]

# Training loop
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0

    for batch_start in range(0, len(train_data), batch_size):
        optimizer.zero_grad()

        # Prepare batch
        batch_data = train_data[batch_start:batch_start + batch_size]
        batch_texts, batch_labels = zip(*batch_data)
        actual_batch_size = len(batch_data)

        # Tokenize and preprocess the texts
        input_ids = torch.randint(0, vocab_size, (actual_batch_size, max_len))
        segment_ids = torch.randint(0, 2, (actual_batch_size, max_len))

        # Convert labels to tensor
        label_tensor = torch.tensor(batch_labels, dtype=torch.long)

        # Forward pass
        output = bert_model(input_ids, segment_ids)

        # Reshape output and labels
        output = output.view(-1, output.shape[-1])  # Reshape to (batch_size * max_len, d_model)
        label_tensor = label_tensor.repeat(max_len)  # Repeat labels for each position

        # Calculate the loss
        loss = criterion(output, label_tensor)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

print(output.shape)    