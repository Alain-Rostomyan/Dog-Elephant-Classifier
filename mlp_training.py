import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# MLP : multi layer perceptron
class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size, hidden_size, output_size = 64*64*3, 64*64*3, 2

# Model initialization
model = MLP1(input_size, hidden_size, output_size)

# The loss function
criterion = nn.CrossEntropyLoss()

# Our optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001) # lr = learning rate


# Define custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder("./dataset", transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

# Define transformation to be applied to images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Create custom dataset instances for training and validation
train_dataset = CustomDataset(root_dir='dataset', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

# Create DataLoader instances for training and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Lists to store training and validation errors
train_errors = []
val_errors = []

# Training loop
min_loss = 0.45
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()  # Set the model to training mode
    total_train_loss = 0.0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images.view(images.size(0), -1))
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_errors.append(avg_train_loss)

    # Validation
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images.view(images.size(0), -1))
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss = total_val_loss / len(valid_loader)
    val_errors.append(avg_val_loss)

    if avg_val_loss < min_loss:
        torch.save(model.state_dict(), 'mlp2_model.pth')
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Plotting the training and validation errors
plt.plot(range(1, num_epochs+1), train_errors, label='Training Error')
plt.plot(range(1, num_epochs+1), val_errors, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training and Validation Errors')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'mlp_model.pth')

# Accessing the model's parameters
for name, param in model.named_parameters():
    print(f'Parameter name: {name}')
    print(f'Parameter shape: {param.shape}')
    print(f'Parameter values: {param}')
