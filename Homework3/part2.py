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

# Defining the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        #This line takes the input tensor x, which contains indices of characters, and passes it through an embedding layer (self.embedding). 
        #The embedding layer converts these indices into dense vectors of fixed size. 
        #These vectors are learned during training and can capture semantic similarities between characters. 
        #The result is a higher-dimensional representation of the input sequence, where each character index is replaced by its corresponding embedding vector. 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=3, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        embedded = self.embedding(x)
        #The RNN layer returns two outputs: 
        #1- the output tensor containing the output of the RNN at each time step for each sequence in the batch, 
        #2-the hidden state (_) of the last time step (which is not used in this line, hence the underscore).
        output, _ = self.rnn(embedded)
        #The RNN's output contains the outputs for every time step, 
        #but for this task, we're only interested in the output of the last time step because we're predicting the next character after the sequence. 
        #output[:, -1, :] selects the last time step's output for every sequence in the batch (-1 indexes the last item in Python).
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

# Hyperparameters
hidden_size = 128
learning_rate = 0.001 
epochs = 20

# Model, loss, and optimizer
model = CharRNN(len(chars), hidden_size, len(chars)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = []
val_loss_list = []

start_time = time.time()

# Training the model
for epoch in range(epochs):

    model.train()
    train_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device, dtype=torch.long), targets.to(device, dtype=torch.long)
        optimizer.zero_grad()  # Clear existing gradients
        outputs = model(sequences)  # Forward pass
        loss = loss_fn(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update model parameters
        train_loss += loss.item() * sequences.size(0)  # Accumulate the loss

    # Calculate average training loss
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)


    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device, dtype=torch.long), targets.to(device, dtype=torch.long)
            val_output = model(sequences)
            loss = loss_fn(val_output, targets)
            val_loss += loss.item() * sequences.size(0)
            #The use of the underscore _ is a common Python convention to indicate that the actual maximum values returned by torch.max are not needed and can be disregarded. 
            #What we are interested in is the indices of these maximum values, which are captured by the variable predicted. These indices represent the model's predictions for each example in the validation set.
            _, predicted = torch.max(val_output, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            #val_accuracy = (predicted == targets).float().mean()
    
    val_loss /= len(test_loader.dataset)
    val_accuracy = correct / total 
 
    #if (epoch+1) % 10 == 0:
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

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

