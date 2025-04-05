import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text  # This is the entire text data

print("Done Download")

# Step 2: Prepare the dataset
sequence_length = 20
# Create a character mapping to integers
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
encoded_text = [char_to_int[ch] for ch in text]

# Create sequences and targets
sequences = []
targets = []
for i in range(0, len(encoded_text) - sequence_length):
    seq = encoded_text[i:i+sequence_length]
    target = encoded_text[i+sequence_length]
    sequences.append(seq)
    targets.append(target)

# Convert lists to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

# Step 3: Create a dataset class
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

# Instantiate the dataset
dataset = CharDataset(sequences, targets)

# Step 4: Create data loaders
batch_size = 128
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

# Defining the Transformer model
class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output)
        return output  # Apply softmax to the linear layer output
     
# Hyperparameters
hidden_size = 128  
num_layers = 4
nhead = 4 
learning_rate = 0.005
epochs = 20

# Model, loss, and optimizer
model = CharTransformer(len(chars), hidden_size, len(chars), num_layers, nhead).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

# Training the model 
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for X_train, Y_train in train_loader:
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        optimizer.zero_grad()
        output = model(X_train)
        output_last = output[:, -1, :]
        loss = criterion(output_last, Y_train)  # Reshape output to match the CrossEntropyLoss expectations
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_train.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_test, Y_test in test_loader:
            X_test, Y_test = X_test.to(device), Y_test.to(device)
            output = model(X_test)
            output_last = output[:, -1, :]
            loss_val = criterion(output_last, Y_test)  # Reshape output to match the CrossEntropyLoss expectations
            running_val_loss += loss_val.item() * X_test.size(0)

            # Calculate accuracy
            _, predicted = torch.max(output_last, dim=1)
            correct += (predicted == Y_test).sum().item()
            total += Y_test.size(0)

    val_loss = running_val_loss / len(test_loader.dataset)
    val_accuracy = correct / total

    print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

end_time = time.time()
execution_time = end_time - start_time
print(f'Total training time: {execution_time:.2f} seconds')

# Computational complexity
n = train_size 
m = sequence_length   
h = hidden_size   
complexity = n * m * h**2
print(f'Approximate computational complexity: O({complexity})')

# Model size complexity
param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model size: {param_size} parameters')