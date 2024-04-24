import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


from model import AlexNet, LeNet

## Test learning to reweight model
# from main import Reweighting


# torch.backends.cudnn.enabled = False
# torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

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
    
    def paper_train(self):
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

            self.gradient_network = AlexNet()
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
                curr_accuracy = self.test()
                print("Epoch " + str(epoch) + " accuracy = ", curr_accuracy, flush=True)

    def train(self):
        # Train the network
        for epoch in range(self.hyperparameters['n_epochs']):
            self.network.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(device)
                target = target.to(device)

                self.optimizer.zero_grad()
                output = self.network(data)
                # print(output)
                # print(target.float())
                loss = self.criterion_mean(output, target.float())
                # print(loss)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.hyperparameters['log_interval'] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()), flush=True)


    def test(self):
        self.network.eval()

        acc = np.array([])
        for itr,(test_img, test_label) in enumerate(self.test_loader):
            prediction = self.network(test_img.to(device)).detach().cpu()
            # print(prediction)
            prediction = (torch.sigmoid(prediction) > 0.5).int().numpy()
            tmp = (prediction == test_label.detach().numpy())
            # print(prediction)
            # print(test_label)
            # print(tmp)
            # acc.append(tmp)
            acc = np.append(acc,tmp)

        # print(acc)
        accuracy = np.mean(acc)
        return np.round(accuracy*100,2)


# train_folder = '../dataset-ninja/train_unbiased'
# test_folder = '../dataset-ninja/test_unbiased'
# validate_folder = '../dataset-ninja/validate_unbiased'
train_folder = '/home/obasit/50024/last_checkpoint/DL-reweighting/Last_checkpoint/dataset/extracted/train_unbiased'
test_folder = '/home/obasit/50024/last_checkpoint/DL-reweighting/Last_checkpoint/dataset/extracted/test_unbiased'
validate_folder = '/home/obasit/50024/last_checkpoint/DL-reweighting/Last_checkpoint/dataset/extracted/validate_unbiased'


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[90.2867748375, 88.24014045, 94.9276596], std=[46.843379357493816, 46.59302413225906, 47.86539694070785]),  # Normalize images
])

train_dataset = datasets.ImageFolder(train_folder, transform=transform)

test_dataset = datasets.ImageFolder(test_folder, transform=transform)

validate_dataset = datasets.ImageFolder(validate_folder, transform=transform)


# number of epoch and log interval reduced for testing
hyperparameters = {
    'n_epochs' : 250,
    'batch_size' : 10000,
    'learning_rate' : 1e-3,
    'momentum' : 0.5,
    'log_interval' : 5
}

network = AlexNet()

# criterion = nn.CrossEntropyLoss(reduction='none')
# criterion_mean = nn.CrossEntropyLoss(reduction='mean')
criterion = nn.BCEWithLogitsLoss(reduction='none')
criterion_mean = nn.BCEWithLogitsLoss(reduction='mean')

optimizer = optim.SGD(network.params(),
                        lr=hyperparameters['learning_rate'],
                        momentum=hyperparameters['momentum'])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=14)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
valid_loader = DataLoader(validate_dataset, batch_size=1000, shuffle=True)

our_model = Reweighting(network, hyperparameters, criterion, criterion_mean, optimizer, train_loader, valid_loader, test_loader)

print("paper", flush=True)
start_accuracy = our_model.test()
print("Starting accuracy = ", start_accuracy, flush=True)

our_model.paper_train()

end_accuracy = our_model.test()
print("Ending accuracy = ", end_accuracy, flush=True)

