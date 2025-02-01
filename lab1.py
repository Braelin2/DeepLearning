import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data_path = './data'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True) 
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

to_tensor = transforms.ToTensor()

transformed_cifar10_train = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))

transformed_cifar10_test = datasets.CIFAR10(data_path, train = False, download = False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))

train_loader = torch.utils.data.DataLoader(transformed_cifar10_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(transformed_cifar10_test, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  

    def forward(self, x):
        x = x.view(-1, 3*32*32)  
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x))  
        x = self.fc4(x)  
        return x

class MLP_Wide(nn.Module):
    def __init__(self):
        super(MLP_Wide, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 1024)  
        self.fc2 = nn.Linear(1024, 512) 
        self.fc3 = nn.Linear(512, 256)  
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP_Deep(nn.Module):
    def __init__(self):
        super(MLP_Deep, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 512) 
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)  
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class MLP_WideDeep(nn.Module):
    def __init__(self):
        super(MLP_WideDeep, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 1024)  
        self.fc2 = nn.Linear(1024, 1024)  
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)  
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)  
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 10)  

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x) 
        return x        

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001) 

n_epochs = 20


train_loss_list = []
val_loss_list = []

for epoch in range(n_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Clear existing gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update model parameters
        train_loss += loss.item() * inputs.size(0)  # Accumulate the loss

    # Calculate average training loss
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, targets)  # Compute loss
            val_loss += loss.item() * inputs.size(0)  # Accumulate the loss

    # Calculate average validation loss (MSE) and RMSE
    val_loss /= len(test_loader.dataset)
    val_loss_list.append(val_loss)

    # Print training and validation results
    print(f'Epoch[{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Plotting training and validation loss
plt.plot(train_loss_list, label='Training Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./imgs/cifar10_loss_plot.png')
plt.show()

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate and print the accuracy
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Calculate and print precision, recall, and F1 score
precision = precision_score(all_targets, all_predictions, average='macro')
recall = recall_score(all_targets, all_predictions, average='macro')
f1 = f1_score(all_targets, all_predictions, average='macro')
print(f'Precision: {precision:.4f}') 
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Generate confusion matrix
conf_matrix = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=transformed_cifar10_train.classes, yticklabels=transformed_cifar10_train.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('./imgs/cf_plot.png')
plt.show()

# Save the model weights
torch.save(model.state_dict(), './models/cifar10.pth')  # .pth is the recommended extension