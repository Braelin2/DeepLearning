import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

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
# part of the data preprocessing step for a character-level text modeling task. 
# Create mappings between characters in the text and numerical indices

#set(text): Creates a set of unique characters found in the text. The set function removes any duplicate characters.
#list(set(text)): Converts the set back into a list so that it can be sorted. 
# sorted(list(set(text))): Sorts the list of unique characters. 
chars = sorted(list(set(text)))
#This line creates a dictionary that maps each character to a unique index (integer)."
ix_to_char = {i: ch for i, ch in enumerate(chars)}
#Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
char_to_ix = {ch: i for i, ch in enumerate(chars)} 
chars = sorted(list(set(text)))




# Preparing the dataset
max_length = 30  # Maximum length of input sequences
X = []
y = []
for i in range(len(text) - max_length):
    sequence = text[i:i + max_length]
    label = text[i + max_length]
    X.append([char_to_ix[char] for char in sequence])
    y.append(char_to_ix[label])

X = np.array(X)
y = np.array(y)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

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
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
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
hidden_size = 256
learning_rate = 0.001
epochs = 100

# Model, loss, and optimizer
model = CharRNN(len(chars), hidden_size, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

# Training the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        #The use of the underscore _ is a common Python convention to indicate that the actual maximum values returned by torch.max are not needed and can be disregarded. 
        #What we are interested in is the indices of these maximum values, which are captured by the variable predicted. These indices represent the model's predictions for each example in the validation set.
        _, predicted = torch.max(val_output, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

end_time = time.time()
execution_time = end_time - start_time
print(f'Total training time: {execution_time:.2f} seconds')

# Computational complexity
n = len(X_train)  
m = max_length    
h = hidden_size   
complexity = n * m * h**2
print(f'Approximate computational complexity: O({complexity})')

# Model size complexity
param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model size: {param_size} parameters')

# Prediction function
def predict_next_char(model, char_to_ix, ix_to_char, initial_str):
    model.eval()
    with torch.no_grad():
        initial_input = torch.tensor([char_to_ix[c] for c in initial_str[-max_length:]], dtype=torch.long).unsqueeze(0)
        prediction = model(initial_input)
        predicted_index = torch.argmax(prediction, dim=1).item()
        return ix_to_char[predicted_index]

# Predicting the next character
test_str = "This is a simple example to demonstrate how to predict the next char"
predicted_char = predict_next_char(model, char_to_ix, ix_to_char, test_str)
print(f"Predicted next character: '{predicted_char}'")