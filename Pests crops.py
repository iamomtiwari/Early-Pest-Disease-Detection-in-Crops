#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


get_ipython().system('pip install kaggle')


# Importing dataset

# In[3]:


import os
# Create the .kaggle directory in the current working directory
os.makedirs('.kaggle', exist_ok=True)
# Move the kaggle.json file to the correct location
get_ipython().system('cp kaggle.json .kaggle/')


# In[4]:


os.chmod('.kaggle/kaggle.json', 0o600)


# In[5]:


#!/bin/bash
get_ipython().system('kaggle datasets download kamal01/top-agriculture-crop-disease')


# In[6]:


import zipfile
import os

# Define the path to the zip file and the extraction directory
zip_file_path = '/home/Tiwari_ME/top-agriculture-crop-disease.zip'
extraction_path = '/home/Tiwari_ME/top-agriculture-crop-disease/'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_path, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

print(f"Dataset extracted to {extraction_path}")


# # Analysing Dataset

# In[7]:


import os

# List the files and directories in the extraction path
extraction_path = '/home/Tiwari_ME/top-agriculture-crop-disease/'
files = os.listdir(extraction_path)

print("Files and directories in the dataset:")
for file in files:
    print(file)


# In[8]:


# Define the path to the 'Crop Diseases' directory
crop_diseases_path = os.path.join(extraction_path, 'Crop Diseases')

# List the files and directories in the 'Crop Diseases' directory
crop_diseases_files = os.listdir(crop_diseases_path)

print("Files and directories in 'Crop Diseases':")
for item in crop_diseases_files:
    print(item)


# # Data Cleaning

# In[9]:


import os
import shutil

# Define the path to the 'Crop Diseases' directory
crop_diseases_path = os.path.join(extraction_path, 'Crop Diseases')

# List of potato-related folders to remove
potato_folders = [
    'Potato___Early_Blight',
    'Potato___Healthy',
    'Potato___Late_Blight'
]

# Remove the potato folders
for folder in potato_folders:
    folder_path = os.path.join(crop_diseases_path, folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # This will remove the folder and its contents
        print(f"Removed folder: {folder_path}")

print("All potato-related folders have been removed.")


# In[10]:


# Verify the remaining folders in 'Crop Diseases'
remaining_folders = os.listdir(crop_diseases_path)
print("\nRemaining folders in 'Crop Diseases':")
for folder in remaining_folders:
    print(folder)


# # Visualising New Dataset

# In[11]:


# Install the necessary libraries
get_ipython().system('pip install matplotlib')

# Import the necessary libraries
import os
import matplotlib.pyplot as plt

# Define the path to the 'Crop Diseases' directory
crop_diseases_path = os.path.join(extraction_path, 'Crop Diseases')

# Dictionary to store the number of images in each folder
image_counts = {}

# Count the number of images in each remaining folder
for folder in os.listdir(crop_diseases_path):
    folder_path = os.path.join(crop_diseases_path, folder)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))  # Count the number of images
        image_counts[folder] = num_images

# Print the counts for verification
print("\nNumber of images in each folder:")
for folder, count in image_counts.items():
    print(f"{folder}: {count} images")

# Step 2: Visualize the counts using a bar plot
plt.figure(figsize=(12, 6))
plt.bar(image_counts.keys(), image_counts.values(), color='skyblue')
plt.xlabel('Disease Categories')
plt.ylabel('Number of Images')
plt.title('Number of Images in Each Crop Disease Category')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()


# In[13]:


get_ipython().system('pip install pytorch-lightning')


# In[14]:


import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Assuming 'crop_diseases_path' is the directory after removing the potato folders
crop_diseases_path = os.path.join(extraction_path, 'Crop Diseases')

# Class labels (excluding potato classes)
class_names = [
    "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Northern_Leaf_Blight",
    "Rice___Brown_Spot", "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast",
    "Sugarcane_Bacterial Blight", "Sugarcane_Healthy", "Sugarcane_Red Rot",
    "Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"
]

# Transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 (adjust as needed)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Load the dataset from the directory (after potato folders were removed)
train_dataset = datasets.ImageFolder(root=crop_diseases_path, transform=transform)

# Check if labels are assigned correctly
print(f"Class-to-index mapping: {train_dataset.class_to_idx}")

# DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Inspecting dataset size
print(f"Number of samples: {len(train_dataset)}")


# In[15]:


class_info = {
    "Corn___Common_Rust": "Apply fungicides as soon as symptoms are noticed. Practice crop rotation and remove infected plants.",
    "Corn___Gray_Leaf_Spot": "Rotate crops to non-host plants, apply resistant varieties, and use fungicides as needed.",
    "Corn___Healthy": "Continue good agricultural practices: ensure proper irrigation, nutrient supply, and monitor for pests.",
    "Corn___Northern_Leaf_Blight": "Remove and destroy infected plant debris, apply fungicides, and rotate crops.",
    "Rice___Brown_Spot": "Use resistant varieties, improve field drainage, and apply fungicides if necessary.",
    "Rice___Healthy": "Maintain proper irrigation, fertilization, and pest control measures.",
    "Rice___Leaf_Blast": "Use resistant varieties, apply fungicides during high-risk periods, and practice good field management.",
    "Rice___Neck_Blast": "Plant resistant varieties, improve nutrient management, and apply fungicides if symptoms appear.",
    "Wheat___Brown_Rust": "Apply fungicides and practice crop rotation with non-host crops.",
    "Wheat___Healthy": "Continue with good management practices, including proper fertilization and weed control.",
    "Wheat___Yellow_Rust": "Use resistant varieties, apply fungicides, and rotate crops.",
    "Sugarcane__Red_Rot": "Plant resistant varieties and ensure good drainage.",
    "Sugarcane__Healthy": "Maintain healthy soil conditions and proper irrigation.",
    "Sugarcane__Bacterial Blight": "Use disease-free planting material, practice crop rotation, and destroy infected plants."
}


# # Data Augmentation 

# In[16]:


get_ipython().system('pip install torch torchvision')


# In[17]:


import os
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the pixel size for resizing
image_size = (224, 224)  # Resize images to 224x224 pixels

from torchvision import transforms

# Define data transformations including augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Define the path to the 'Crop Diseases' directory
crop_diseases_path = os.path.join(extraction_path, 'Crop Diseases')

# Create a dataset
dataset = datasets.ImageFolder(root=crop_diseases_path, transform=data_transforms)

# Create a DataLoader
batch_size = 32  # Define your batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example: Visualizing some augmented images
def visualize_augmented_images(dataloader):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Convert tensor images back to numpy for visualization
    images = images.numpy().transpose((0, 2, 3, 1))  # Change the shape to (batch_size, height, width, channels)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    for i in range(8):  # Show 8 images
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# Visualize the augmented images
visualize_augmented_images(dataloader)


# # Oversampling

# In[18]:


import os
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

# Define the path to the 'Crop Diseases' directory
crop_diseases_path = os.path.join(extraction_path, 'Crop Diseases')

# Create a dataset (assuming your augmentation is already applied elsewhere)
dataset = datasets.ImageFolder(root=crop_diseases_path)

# Count images in each class
class_counts = Counter(dataset.targets)  # dataset.targets contains class indices
max_count = max(class_counts.values())  # Find the maximum count

# Prepare weights for each class for oversampling
weights = []
for target in dataset.targets:
    weights.append(max_count / class_counts[target])  # Calculate weight for oversampling

# Create a WeightedRandomSampler
sampler = WeightedRandomSampler(weights, num_samples=max_count * len(class_counts), replacement=True)

# Create a DataLoader with the sampler
batch_size = 32  # Define your batch size
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Now you can use the dataloader for training your model


# In[19]:


from collections import Counter
# Get the indices of the samples in the dataloader
sample_indices = [i for i, _ in enumerate(dataloader.dataset)]
# Count how many samples correspond to each class after oversampling
oversampled_class_counts = Counter(dataloader.dataset.targets[i] for i in sample_indices)
# Display the oversampled class counts
# Display the maximum count per class after oversampling
print("Number of images in each class after oversampling:")
for class_index in range(len(class_counts)):
    print(f"Class {class_index}: {max_count} images")


# In[20]:


import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

# Set the path to the dataset
data_dir = '/home/Tiwari_ME/top-agriculture-crop-disease/Crop Diseases'

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# Count images in each class
class_counts = Counter(dataset.targets)
max_count = 1488  # We want all classes to have 1488 images

# Prepare weights for each class to perform oversampling
weights = []
for target in dataset.targets:
    weights.append(max_count / class_counts[target])

# Create a WeightedRandomSampler for oversampling
sampler = WeightedRandomSampler(weights, num_samples=max_count * len(class_counts), replacement=True)

# Create a DataLoader with the sampler for oversampling
batch_size = 32
oversampled_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Confirm the new class distribution after oversampling
sample_indices = [i for i, _ in enumerate(oversampled_dataloader.dataset)]
oversampled_class_counts = Counter(oversampled_dataloader.dataset.targets[i] for i in sample_indices)
print("Number of images in each class after oversampling:")
for class_index in range(len(class_counts)):
    print(f"Class {class_index}: {max_count} images")


# In[21]:


total_oversampled_images = max_count * len(class_counts)
print(f"Total images after oversampling: {total_oversampled_images}")


# # Model Training

# In[22]:


get_ipython().system('pip install torch torchvision transformers')


# In[23]:


get_ipython().system('pip install transformers datasets')
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# In[25]:


import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Number of classes in your dataset
num_classes = 14

# Load the pre-trained ViT model and replace the classifier head
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes
)

# Load the feature extractor for pre-processing
feature_extractor = ViTFeatureExtractor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)

# Check if a GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# In[30]:


from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load pre-trained ViT model with a modified number of labels
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(class_counts),  # Use your actual number of classes
    ignore_mismatched_sizes=True  # Ignore size mismatches in the classifier layer
)


# In[31]:


import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load pre-trained ViT model with the correct number of labels and ignore mismatched sizes
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(class_counts),  # Adjust this to the number of classes in your dataset
    ignore_mismatched_sizes=True  # Allow for size mismatches in the classifier layer
)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[32]:


from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define data transformations using the feature extractor
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ViT input size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Load the dataset
data_dir = crop_diseases_path  # Your dataset path
dataset = ImageFolder(root=data_dir, transform=data_transforms)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


# In[33]:


import torch
from torch import optim
from tqdm import tqdm  # For progress bar

# Set your model to training mode
model.train()

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # You can adjust the learning rate
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5  # Define the number of epochs you want
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in tqdm(dataloader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images).logits  # Get logits
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


# In[35]:


import torch

# Assuming `model` is your trained ViT model
model_save_path = 'vit_model.pth'

# Save the entire model
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")


# In[72]:


# Assuming 'model' is your trained model
torch.save(model.state_dict(), r"G:\crop_disease_prediction\model\vit_model.pth")


# In[43]:


import torch
import torchvision.models as models  # If using a pretrained model


# In[45]:


import torchvision
print(torchvision.__version__)


# In[46]:


get_ipython().system('pip install timm')


# In[49]:


get_ipython().system('pip install timm torch torchvision')


# In[50]:


import timm

# Define the Vision Transformer model (using timm)
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=14)


# In[58]:


import torch

# Load the entire model
model = torch.load('vit_model.pth', map_location='cpu')

# Set the model to evaluation mode
model.eval()


# In[59]:


import torch

# Check the contents of the file
try:
    state_dict = torch.load('vit_model.pth', map_location='cpu')
    print("Loaded successfully!")
except Exception as e:
    print(f"Error loading the file: {e}")


# In[60]:


model.eval()


# In[61]:


from torchvision import transforms
from PIL import Image

# Define the image transformations (adjust parameters as needed)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Use the mean and std used during training
])

# Load an image (replace 'path_to_your_image' with the actual path)
img = Image.open('wheat yellow rust.jpg')
img_t = preprocess(img)
batch_t = img_t.unsqueeze(0)  # Add a batch dimension


# In[63]:


with torch.no_grad():  # No need to compute gradients during inference
    output = model(batch_t)

# Extract logits from the output object
logits = output.logits  # This is the correct way to access the logits

# Get the predicted class (assuming logits are used)
_, predicted = torch.max(logits, dim=1)  # Use 'dim=1' to find the class with the maximum score
predicted_class = predicted.item()

# Map predicted class to solution as before
predicted_class_name = class_names[predicted_class]
solution = class_info[predicted_class_name]

print(f"Predicted Class: {predicted_class_name}")
print(f"Recommended Action: {solution}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


for param in model.vit.parameters():
    param.requires_grad = False  # Freeze the ViT base layers

# Fine-tune the classifier
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)


# # Transfer Learning

# In[37]:


from transformers import ViTForImageClassification
import torch

# Load pre-trained ViT model and ignore mismatched sizes
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(dataset.classes),
    ignore_mismatched_sizes=True
)

# Resize the classifier to match the number of classes in your dataset
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=len(dataset.classes))

# Move model to device (GPU/CPU)
model.to(device)

# Continue with training or evaluation as needed


# # Fine Tuning 

# In[38]:


import torch
from torch import nn, optim
from tqdm import tqdm

# Define the optimizer, loss function, and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # AdamW is commonly used for ViT
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification

# Fine-tuning the model
num_epochs = 5
train_loss = []
train_accuracy = []

model.train()  # Set model to training mode

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Loop over batches of data
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).logits  # Get logits from the model
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    # Compute epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions * 100
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Save the fine-tuned model
torch.save(model.state_dict(), 'finetuned_vit_model.pth')
print("Model saved as 'finetuned_vit_model.pth'")


# In[70]:


from transformers import ViTForImageClassification, ViTFeatureExtractor

# Specify the model you want to download
model_name = "google/vit-base-patch16-224"

# Download the model and feature extractor
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Save the model and feature extractor locally
model.save_pretrained(r"G:\crop_disease_prediction")
feature_extractor.save_pretrained(r"G:\crop_disease_prediction")


# In[71]:


import os

os.makedirs(r"G:\crop_disease_prediction", exist_ok=True)


# In[ ]:




