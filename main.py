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

class Reweighting():
    def __init__(self, network, hyperparameters, criterion, criterion_mean, optimizer, train_loader, valid_loader, test_loader):
        self.network = network.requires_grad_(requires_grad=True)
        self.hyperparameters = hyperparameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion_mean = criterion_mean
        self.gradient_network = None

    def train(self):
        # X_g = LeNet5()
        # print("Starting training...")
        X_g, y_g = next(iter(self.valid_loader))
        X_g = X_g.to(device)
        y_g = y_g.to(device)

        y_f_hat = torch.empty(1)
        y_f_hat = y_f_hat.to(device)

        theta_tp1 = self.network.state_dict()
        for epoch in range(self.hyperparameters['n_epochs']):
            self.network.train()

            self.gradient_network = LeNet()
            self.gradient_network.load_state_dict(theta_tp1)

            # get batch of data from train_loader
            X_f, y_f = next(iter(self.train_loader))
            X_f = X_f.to(device)
            y_f = y_f.to(device)

            # Line 4
            y_f_hat = self.gradient_network(X_f)
            
            # Line 5
            epsilon = torch.zeros(y_f.size(), requires_grad=True)
            epsilon = epsilon.to(device)

            Costs = self.criterion(y_f_hat, y_f.float())
            l_f = torch.sum(torch.mul(Costs, epsilon))

            # Line 6
            grad_t = torch.autograd.grad(outputs=l_f, inputs=self.gradient_network.params(), create_graph=True)
            

            # Line 7: manually update the weights of the validation network
            lr = self.hyperparameters['learning_rate']
            self.gradient_network.update_params_SGD_step(lr, grad_t)

            # Line 8
            # Model has theta_hat
            y_g_hat = self.gradient_network(X_g)

            # Line 9
            l_g = self.criterion_mean(y_g_hat, y_g.float())

            # Line 10
            grad_epsilon = torch.autograd.grad(l_g, epsilon, only_inputs=True)[0]

            # Line 11
            w_tilde = torch.clamp(-grad_epsilon, min=0)

            if torch.sum(w_tilde) != 0:
                w = w_tilde / torch.sum(w_tilde)
            else:
                w = w_tilde

            # Line 12
            y_f_hat = self.network(X_f)
            Costs = self.criterion(y_f_hat, y_f.float())
            l_f_hat = torch.sum(torch.mul(Costs, w))

            self.optimizer.zero_grad()

            # Line 13
            l_f_hat.backward()

            # Line 14
            self.optimizer.step()
            # break

            theta_tp1 = self.network.state_dict()

            if (epoch % self.hyperparameters['log_interval'] == 0):
                self.network.eval()

                test_loss = 0
                correct = 0
                total_samples = 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(self.valid_loader):
                        output = self.network(data.to(device)).cpu()
                        batch_size = data.size(0)
                        total_samples += batch_size
                        # test_loss += self.criterion(output, target.float()).item()
                        pred = (torch.sigmoid(output) > 0.5).int()
                        if (pred.size(dim=0) != 10): # edge case with batch size
                            continue
                        correct += (pred == target.int()).sum().item()

                    accuracy = correct / total_samples
                    # print("validation accuracy ", accuracy)

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
                # test_loss += self.criterion(output, target.float()).item()
                pred = (torch.sigmoid(output) > 0.5).int()
                correct += (pred == target.int()).sum().item()

        return correct / total_samples  

hyperparameters = {
    'n_epochs' : 500,
    'batch_size_train' : 10000,
    'batch_size_valid' : 100,
    'batch_size_test' : 100,
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
    # # Load the data
    # data_loader = MNISTDataLoader(validation_ratio=0.05,
    #                             batch_size_train=hyperparameters['batch_size_train'],
    #                             batch_size_valid=hyperparameters['batch_size_valid'],
    #                             batch_size_test=hyperparameters['batch_size_test'])

    # desired_sample_distribution = [100, perc]
    # data_loader.sample_bias(desired_sample_distribution, dataset="train")

    # train_loader = data_loader.train_dataloader
    # valid_loader = data_loader.valid_dataloader
    # test_loader = data_loader.test_dataloader

    train_folder = './dataset-ninja/train_binary'
    test_folder = './dataset-ninja/test_binary'
    validate_folder = './dataset-ninja/validate_binary'

    class_weights = [0.5, 0.5]  # Example weights for each class
    # class_weights = [804/161429, 160625/161429]  # Oversampling weights
    # Train count:  {'car': 160625, 'motorcycle': 804}

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=(1/2.903307641932519,), std=(0.17295126362098218,)),  # Normalize images
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

    our_model = Reweighting(network, hyperparameters, criterion, criterion_mean, optimizer, train_loader, valid_loader, test_loader)

    # Starting accuracy
    accuracy = our_model.test()
    print("Starting accuracy: ", accuracy)

    our_model.train()

    # Ending accuracy
    accuracy = our_model.test()
    print("Current accuracy: ", accuracy)

    acc_arr.append(accuracy)
    # print("testing " + str(perc) + " accuracy = " + str(accuracy))

df["accuracy"] = acc_arr
print(df)
# print(df)
df.to_csv("realworld_accuracy_tsting_learn2reweight.csv")
