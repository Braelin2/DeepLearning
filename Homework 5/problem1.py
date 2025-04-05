import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Sample text
#text = "This is a simple example to demonstrate how to predict the next character using RNN in PyTorch."

# Sample text
text = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character
        in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion,
        spell checking, and even in the development of sophisticated AI models capable of generating human-like text.At its core, next character prediction
        relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow.
        These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.One of the most 
        popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called 
        Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory'
        about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term
        dependencies, making them even more effective for next character prediction tasks. Training a model for next character prediction involves feeding
        it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this
        training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its
        predictive accuracy over time. Once trained, the model can be used to predict the next character in a given piece of text by considering the 
        sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments 
        with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants. In summary, next character 
        prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate,
        and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve,
        opening new possibilities for the future of text-based technology."""

# Creating character vocabulary
chars = sorted(list(set(text)))
ix_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# Preparing the dataset for sequence prediction
X = []
y = []
max_length = 10  # Maximum length of input sequences
for i in range(len(text) - max_length - 1):
    sequence = text[i:i + max_length]
    label_sequence = text[i+1:i + max_length + 1]  # Shift by one for the next character sequence
    X.append([char_to_ix[char] for char in sequence])
    y.append([char_to_ix[char] for char in label_sequence])

X = np.array(X)
y = np.array(y)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

train_size = int(len(X_train))

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

# Defining the Transformer model
class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)  # Softmax layer over the feature dimension

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output)
        return self.softmax(output)  # Apply softmax to the linear layer output

# Hyperparameters
hidden_size = 256
num_layers = 1
nhead = 4
learning_rate = 0.0005
epochs = 50

# Model, loss, and optimizer
model = CharTransformer(len(chars), hidden_size, len(chars), num_layers, nhead)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

# Training the model
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.transpose(1, 2), y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output.transpose(1, 2), y_val)  # Same transpose for validation
        _, predicted = torch.max(val_output, 2)  # Adjust dimension for prediction
        val_accuracy = (predicted == y_val).float().mean()  # Calculate accuracy

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction[0, -1], dim=0).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "This is a simple example to demonstrate how to predict the next char"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")

end_time = time.time()
execution_time = end_time - start_time
print(f'Total training time: {execution_time:.2f} seconds')

# Computational complexity
n = train_size 
m = max_length   
h = hidden_size   
complexity = n * m * h**2
print(f'Approximate computational complexity: O({complexity})')

# Model size complexity
param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model size: {param_size} parameters')