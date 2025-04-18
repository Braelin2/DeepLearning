import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from thop import profile
import time
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_txt = []

# Hyperparameters
image_size = 224
num_classes = 100
num_epochs = 10
batch_size = 64
learning_rate = 0.001


# Data preparation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        all_txt.append("Epoch [" + str(epoch+1) + "/" + str(num_epochs) + "], Loss: " + str(loss.item()))
        stop_time = time.time()
        total_time = stop_time - start_time
        print(f"Epoch {epoch+1} training time: {total_time:.2f} seconds")
        all_txt.append("Epoch" + str(epoch+1) +  " training time: " + str(total_time) + " seconds")

# Test the model
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy: {100 * correct / total}%')
        all_txt.append("Accuracy: " + str(100 * correct / total) + "%")

# Run training and testing
if __name__ == '__main__':

    # Initialize a pretrained ResNet-18 model
    model = models.resnet18(pretrained=True)
    # Modify the final fully connected layer to output 100 classes for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training started...")
    all_txt.append("Training started...")
    train()
    
    print("\nTesting started...")
    all_txt.append("Testing started...")
    test()
    
    # Print model summary with the new input size
    summary_str = str(summary(model, input_size=(batch_size, 3, 224, 224)))
    print(summary_str)
    all_txt.append(summary_str)
    
    # Save logs to file
    file_path = "problem1_resnet.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        for line in all_txt:
            file.write(line + "\n")
