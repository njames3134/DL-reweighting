import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import LeNet5
from meta_model import MetaModule, LeNet
from dataloader_weightedsample import MNISTDataLoaderWeightedSample 

torch.backends.cudnn.enabled = False
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NoReweighting():
    def __init__(self, network, hyperparameters, criterion, optimizer, train_loader, test_loader):
        self.network = network
        self.hyperparameters = hyperparameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_loader
        self.test_loader = test_loader

    def train(self):
        # Train the network
        for epoch in range(hyperparameters['n_epochs']):
            self.network.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                self.optimizer.zero_grad()
                output = self.network(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # if batch_idx % hyperparameters['log_interval'] == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item()))


    def test(self):
        test_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            self.network.eval()
            output = self.network(data.to(device)).cpu()
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        # print('\nTest set: \nAvg. loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100.0 * correct / len(test_loader.dataset)))
        
        return (100.0 * correct / len(self.test_loader.dataset)).item()

hyperparameters = {
    'n_epochs' : 10000,
    'batch_size_train' : 100,
    'batch_size_valid' : 10,
    'batch_size_test' : 1000,
    'learning_rate' : 1e-3,
    'momentum' : 0.5,
    'log_interval' : 500
}

avging_size = 2
# perc_9_arr = [100, 25, 10, 5, 1, 0.5]
perc_9_arr = [1, 0.5]

df = pd.DataFrame(columns=[str(x) for x in perc_9_arr])

for perc in perc_9_arr:
    print(perc) 
    acc_arr = []
    for repeat in range(avging_size):
        network = LeNet(n_out=10)

        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion_mean = nn.CrossEntropyLoss(reduction='mean')

        optimizer = optim.SGD(network.params(),
                                lr=hyperparameters['learning_rate'],
                                momentum=hyperparameters['momentum'])
        # Load the data
        desired_sample_distribution = [100, 100, 100, 100, 100, 100, 100, 100, 100, perc]
        train_sample_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1/(perc/100)]
        data_loader = MNISTDataLoaderWeightedSample(desired_sample_distribution,
                                            train_sample_weights,
                                            validation_ratio=0.05,
                                            batch_size_train=hyperparameters['batch_size_train'],
                                            batch_size_valid=hyperparameters['batch_size_valid'],
                                            batch_size_test=hyperparameters['batch_size_test'],
                                            shuffle=True)


        train_loader = data_loader.train_dataloader
        valid_loader = data_loader.valid_dataloader
        test_loader = data_loader.test_dataloader


        our_model = NoReweighting(network, hyperparameters, criterion_mean, optimizer, train_loader, test_loader)

        our_model.train()
        accuracy = our_model.test()
        acc_arr.append(accuracy)
        # print("testing " + str(perc) + " accuracy = " + str(accuracy))
    df[str(perc)] = acc_arr
    print(df)
print(df)
df.to_csv("accuracy_tsting_no_weights_resmapled1.csv")
