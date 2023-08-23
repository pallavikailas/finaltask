import torch
import torch.nn as nn
from bert import BERT

sequence_length = 10  
feature_dim = 768    

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layer for the output (classification)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)
        
        # Apply softmax activation to each time step
        out = torch.softmax(out, dim=2)  # Apply softmax along the second dimension (classes)

        return out


hidden_size  = 64
num_layers = 1
num_classes = 2

bert_model = BertModel.from_pretrained(bert)

# Create the BiLSTM model
input_size = feature_dim  # Input size should match the BERT output dimension
bert = BiLSTM(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bert.parameters(), lr=0.001)

# Training loop
batch_size = 32
epochs = 10
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        # Forward pass through BERT
        with torch.no_grad():  # Disable gradient computation for BERT
            bert_output = bert_model(inputs)

        # Use BERT output as input for the BiLSTM
        lstm_input = bert_output.last_hidden_state  # Use BERT output as input
        predictions = bert(lstm_input)

        # Reshape the predictions to (batch_size * sequence_length, num_classes)
        predictions = predictions.view(-1, num_classes)

        loss = criterion(predictions, labels)  # Cross-entropy loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
