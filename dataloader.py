import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

class MNISTDataLoader:
    def __init__(self, validation_ratio=0.05, batch_size_train=100, batch_size_valid=100, batch_size_test=1000, shuffle=True):
        self.shuffle = shuffle
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32)), # lenet5 is based on 32x32 images, mnist is 28x28
            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean and STD of MNIST
        ])
        self.train_dataloader = self.__get_MNIST_train_dataloader()
        self.valid_dataloader = self.__get_MNIST_valid_dataloader(validation_ratio)
        self.test_dataloader = self.__get_MNIST_test_dataloader()

    def __get_MNIST_train_dataloader(self):
        # Method that loads in MNIST training data and returns a dataloader
        data_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('.', train=True, download=True, transform = self.transform),
          batch_size=self.batch_size_train, shuffle=self.shuffle
        )
        return data_loader

    def __get_MNIST_valid_dataloader(self, validation_ratio):
        # Method that created a validation dataset using a random sample
        # of training data. Validation ratio is used to determine
        ds = self.__get_dataset("train")

        # Generate list of random indicies to take from the training data
        n = int(len(ds)*validation_ratio)
        sample_index = np.random.choice(range(len(ds)), size=n, replace=False)
        sample_index = np.sort(sample_index)

        # Create data loader and define targets and data from the training dataset
        data_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('.', train=True, download=True, transform = self.transform),
          batch_size=self.batch_size_valid, shuffle=self.shuffle
        )
        data_loader.dataset.targets = ds.targets[sample_index]
        data_loader.dataset.data = ds.data[sample_index]

        # # Remove validation data from training data (No repeats)
        # sample_index_opp = [i for i in range(len(ds)) if i not in sample_index]
        # ds.targets = ds.targets[sample_index_opp]
        # ds.data = ds.data[sample_index_opp]
        
        return data_loader

    def __get_MNIST_test_dataloader(self):
        # Method that loads in MNIST testing data and returns a dataloader
        data_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('.', train=False, download=True, transform = self.transform),
          batch_size=self.batch_size_test, shuffle=self.shuffle
        )
        return data_loader

    def __get_dataset(self, dataset):
        # Support method to return the requested dataset. Also throws and error is user
        # requests and invalid dataset string.
        if dataset.lower() == "train":
            ds = self.train_dataloader.dataset
        elif dataset.lower() == "valid":
            ds = self.valid_dataloader.dataset
        elif dataset.lower() == "test":
            ds = self.test_dataloader.dataset
        else:
            error_msg = "Please provide either 'train' or 'test'."
            raise Exception(error_msg)
        return ds

    def get_num(self, dataset="train"):
        # Returns the number of samples for specified dataset
        ds = self.__get_dataset(dataset)
        return len(ds)

    def get_curr_dist(self, dataset="train"):
        # Returns the current distribution of samples for specified dataset
        # Note that the returned list is the number of each sample, not normalized frequency
        ds = self.__get_dataset(dataset)
        curr_dist = []
        for lcv in range(10):
            curr_dist.append(len(ds.targets[ds.targets == lcv]))
        return curr_dist

    def sample_bias(self, desired_sample_distribution, dataset="train", dist_is_freq=True):
        # Force a sample bias on the specified dataset. Note that samples will be removed in order
        # to enforce provided distribution. The maximum number of samples will be retained however.
        # dist_is_freq: True if provided distribution is a list of frequencies. True if list of absolute number desired.

        ds = self.__get_dataset(dataset)

        # Modify desired distribution vector so that is uses maximum number of samples possible
        if dist_is_freq:
            dist = np.array(desired_sample_distribution)
            dist_freq = dist / np.sum(dist)
            curr_dist = np.array(self.get_curr_dist(dataset=dataset))
            # See which maximum value should be used
            max_num = 0
            while np.all(curr_dist > max_num*dist_freq):
                max_num += 1
            desired_sample_distribution = np.floor(max_num*dist_freq).astype(int)

        # Randomly sample each class to get a final list of sampled indicies
        sample_index = np.empty((0,1), dtype=int)
        for lcv, n in enumerate(desired_sample_distribution):
            curr_idx = np.where(ds.targets == lcv)
            sample_index = np.append(sample_index, np.random.choice(curr_idx[0], size=n, replace=False))
        sample_index = np.sort(sample_index)
        ds.targets = ds.targets[sample_index]
        ds.data = ds.data[sample_index]

    def corrupt_targets(self, start_target, end_target, freq_corrupt, dataset="train"):
        # Not used in current checkpoint, but will corrupt a random selection of start_target class
        # samples and convert their targets to the end_target classification.
        # freq_corrupt: Frequency in which the corruption occurs. 0.2 = 20% corrupted samples.
        ds = self.__get_dataset(dataset)
        assert freq_corrupt <= 1.0

        # Randomly select indicies from start_target class and corrupt targets
        n = int(torch.sum(ds.targets == start_target).item() * freq_corrupt)
        curr_idx = np.where(ds.targets == start_target)
        sample_index = np.random.choice(curr_idx[0], size=n, replace=False)
        ds.targets[sample_index] = end_target

    def plot_dist(self, dataset="train"):
        # Plots the desired dataset distribution
        # Plot is not normalized, meaning the plot shows the number of samples for each class.
        ds = self.__get_dataset(dataset)
        plt.figure()
        plt.hist(ds.targets, bins=range(11), align='left')
        if dataset.lower() == "valid":
            plt.title(dataset.capitalize() + "ation Dataset Distribution")
        else:
            plt.title(dataset.capitalize() + "ing Dataset Distribution")
        plt.xlabel("Handwritten Number")
        plt.ylabel("Frequency")
        plt.xticks(ticks=range(10))
        plt.grid()
