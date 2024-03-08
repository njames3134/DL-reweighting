import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import LeNet
from dataloader import MNISTDataLoader 

torch.backends.cudnn.enabled = False
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            Costs = self.criterion(y_f_hat, y_f)
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
            l_g = self.criterion_mean(y_g_hat, y_g)

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
            Costs = self.criterion(y_f_hat, y_f)
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

                acc = []
                for itr,(test_img, test_label) in enumerate(self.valid_loader):
                    prediction = self.network(test_img.to(device)).detach().cpu().numpy()
                    prediction = np.argmax(prediction, axis=1)
                    # print(prediction)
                    # print(test_label.detach().numpy())
                    tmp = (prediction == test_label.detach().numpy())
                    tmp = tmp*1
                    acc.append(tmp)

                accuracy = np.concatenate(acc).mean()
                # print("validation loss ", np.round(accuracy*100,2))


    def test(self):
        self.network.eval()

        acc = []
        for itr,(test_img, test_label) in enumerate(self.test_loader):
            prediction = self.network(test_img.to(device)).detach().cpu().numpy()
            prediction = np.argmax(prediction, axis=1)
            tmp = (prediction == test_label.detach().numpy())
            tmp = tmp*1
            acc.append(tmp)

        accuracy = np.concatenate(acc).mean()
        return np.round(accuracy*100,2)

hyperparameters = {
    'n_epochs' : 5000,
    'batch_size_train' : 100,
    'batch_size_valid' : 10,
    'batch_size_test' : 1000,
    'learning_rate' : 1e-3,
    'momentum' : 0.5,
    'log_interval' : 500
}

avging_size = 5
perc_9_arr = [100, 25, 10, 5, 1, 0.5]

df = pd.DataFrame(columns=[str(x) for x in perc_9_arr])

for perc in perc_9_arr:
    acc_arr = []
    print(perc)
    for repeat in range(avging_size):
        network = LeNet()

        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion_mean = nn.CrossEntropyLoss(reduction='mean')

        optimizer = optim.SGD(network.params(),
                                lr=hyperparameters['learning_rate'],
                                momentum=hyperparameters['momentum'])
        # Load the data
        data_loader = MNISTDataLoader(validation_ratio=0.05,
                                    batch_size_train=hyperparameters['batch_size_train'],
                                    batch_size_valid=hyperparameters['batch_size_valid'],
                                    batch_size_test=hyperparameters['batch_size_test'])


        desired_sample_distribution = [100, 100, 100, 100, 100, 100, 100, 100, 100, perc]
        data_loader.sample_bias(desired_sample_distribution, dataset="train")


        train_loader = data_loader.train_dataloader
        valid_loader = data_loader.valid_dataloader
        test_loader = data_loader.test_dataloader


        our_model = Reweighting(network, hyperparameters, criterion, criterion_mean, optimizer, train_loader, valid_loader, test_loader)

        our_model.train()
        accuracy = our_model.test()
        acc_arr.append(accuracy)
        # print("testing " + str(perc) + " accuracy = " + str(accuracy))
    df[str(perc)] = acc_arr
    print(df)
print(df)
df.to_csv("accuracy_tsting.csv")
