import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import LeNet
from dataloader import MNISTDataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

torch.backends.cudnn.enabled = False
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_images(dataloader, num_images):
    for batch_idx, (data, target) in enumerate(dataloader):
        for i in range(len(data)):
            plt.imshow(data[i].squeeze(), cmap='gray')
            plt.title(f"Label: {target[i]}")
            plt.axis('off')
            plt.show()

            num_images -= 1
            if num_images == 0:
                return

class NoReweighting():
    def __init__(self, network, hyperparameters, criterion, optimizer, train_loader, test_loader):
        self.network = network
        self.hyperparameters = hyperparameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        # Train the network
        for epoch in range(hyperparameters['n_epochs']):
            self.network.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(device)
                target = target.to(device)

                self.optimizer.zero_grad()
                output = self.network(data)

                loss = self.criterion(output, target.float())
                loss.backward()
                self.optimizer.step()
                if batch_idx % hyperparameters['log_interval'] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))


    def test(self):
        test_loss = 0
        correct = 0
        total_samples = 0

        self.network.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                output = self.network(data.to(device)).cpu()
                batch_size = data.size(0)
                total_samples += batch_size
                test_loss += self.criterion(output, target.float()).item()
                pred = (torch.sigmoid(output) > 0.5).int()
                correct += (pred == target.int()).sum().item()

        return correct / total_samples 

hyperparameters = {
    'n_epochs' : 30,
    'batch_size_train' : 10000,
    'batch_size_valid' : 1000,
    'batch_size_test' : 1000,
    'learning_rate' : 1e-3,
    'momentum' : 0.5,
    'log_interval' : 50
}

avging_size = 5
# perc_9_arr = [100, 25, 10, 5, 1, 0.5]

df = pd.DataFrame(columns=["accuracy"])

# for perc in perc_9_arr:
acc_arr = []
for repeat in range(avging_size):
    network = LeNet()

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    criterion_mean = nn.BCEWithLogitsLoss(reduction='mean')

    optimizer = optim.SGD(network.params(),
                            lr=hyperparameters['learning_rate'],
                            momentum=hyperparameters['momentum'])

    # Load the data
    train_folder = './dataset-ninja/train_binary'
    test_folder = './dataset-ninja/test_binary'
    validate_folder = './dataset-ninja/validate_binary'

    # class_weights = [0.5, 0.5]  # Example weights for each class
    class_weights = [804/161429, 160625/161429]  # Oversampling weights
    # Train count:  {'car': 160625, 'motorcycle': 804}

    # # No weights transform
    # transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.ToTensor(),  # Convert images to PyTorch tensors
    #     transforms.Normalize(mean=(1/2.903307641932519,), std=(0.17295126362098218,)),  # Normalize images
    # ])

    # Resampling transform
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=(1/2.893307641932519,), std=(0.17328706989317727,)),  # Normalize images
    ])

    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    weights = [class_weights[label] for label in train_dataset.targets]
    train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    test_dataset = datasets.ImageFolder(test_folder, transform=transform)

    validate_dataset = datasets.ImageFolder(validate_folder, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size_train'], sampler=train_sampler)
    # train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size_test'], shuffle=True)
    valid_loader = DataLoader(validate_dataset, batch_size=hyperparameters['batch_size_valid'], shuffle=True)

    # show_images(test_loader, 100)

    our_model = NoReweighting(network, hyperparameters, criterion_mean, optimizer, train_loader, test_loader)

    # Starting accuracy
    accuracy = our_model.test()
    print("Starting accuracy: ", accuracy)

    our_model.train()

    # Ending accuracy
    accuracy = our_model.test()
    print("Ending accuracy: ", accuracy)

    acc_arr.append(accuracy)
    # print("testing " + str(perc) + " accuracy = " + str(accuracy))
df["accuracy"] = acc_arr
print(df)
# print(df)
df.to_csv("realworld_accuracy_tsting_oversampling.csv")
