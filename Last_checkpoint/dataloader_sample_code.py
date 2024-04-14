import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_folder = './dataset/train'
test_folder = './dataset/test'
validate_folder = './dataset/validate'

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[119.93560047938638, 121.99074304889741, 129.42976558005753], std=[65.61636024791385, 64.00107977894356, 60.628164585048744]),  # Normalize images
])

# Create ImageFolder datasets
train_dataset = datasets.ImageFolder(train_folder, transform=transform)
test_dataset = datasets.ImageFolder(test_folder, transform=transform)
validate_dataset = datasets.ImageFolder(validate_folder, transform=transform)

# Define batch size
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)


