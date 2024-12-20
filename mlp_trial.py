# Python packages
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

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
# Load the saved model
model = MLP1(input_size, hidden_size, output_size) # model initialization
model.load_state_dict(torch.load('./mlp_model.pth')) # take already prepared weights
model.eval()  # Set the model to evaluation mode

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image to match model input size
    transforms.ToTensor()  # Convert PIL image to PyTorch tensor
])

# Load and preprocess the image
image_path = './dog_test.jpg'  # Specify the path to your image

image = Image.open(image_path)
image = image_transform(image)
image = image.view(1, -1)  # Add batch dimension

# Forward pass through the model to get predictions
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

# Define class labels
class_labels = ['Dog', 'Elephant']

# Print prediction
print(f'The predicted class is: {class_labels[predicted_class]}')
