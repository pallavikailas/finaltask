import torch
import torch.nn as nn

# Define your input sequence length and feature dimension
sequence_length = 10  # Length of each input sequence
feature_dim = 5       # Dimensionality of each feature in the sequence

# Create a class for the Bidirectional LSTM model
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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = feature_dim
hidden_size = 64
num_layers = 1
num_classes = 2  # Change this based on your problem

# Create the BiLSTM model
model = BiLSTM(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate some random training data (replace with your own data)
X_train = torch.rand(100, sequence_length, feature_dim)
y_train = torch.randint(2, (100,), dtype=torch.long)

# Training loop
batch_size = 32
epochs = 10
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Now you can use the trained model to make predictions on new data
# Replace X_test with your test data
X_test = torch.rand(10, sequence_length, feature_dim)
with torch.no_grad():
    predictions = model(X_test)
    _, predicted = torch.max(predictions, 1)

print("Predicted classes:", predicted)
