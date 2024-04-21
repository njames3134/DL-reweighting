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
        # Method that creates a validation dataset using a random sample
        # of training data. Validation ratio is used to determine

        ds = self.__get_dataset("train")

        # Generate list of random indicies to take from the training data
        n = 0
        n += len(ds.targets[ds.targets == 4])
        n += len(ds.targets[ds.targets == 9])
        n *= validation_ratio
        class_labels = [4, 9]

        sample_index = np.empty((0,), dtype=int)
        for label in class_labels:
            curr_idx = np.where(ds.targets == label)[0]
            sampled_indices = np.random.choice(curr_idx, size=int(n), replace=False)
            sample_index = np.concatenate((sample_index, sampled_indices))

        # Create data loader and define targets and data from the training dataset
        data_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('.', train=True, download=True, transform = self.transform),
          batch_size=self.batch_size_valid, shuffle=self.shuffle
        )
        data_loader.dataset.targets = ds.targets[sample_index] == 9
        data_loader.dataset.data = ds.data[sample_index]

        return data_loader

    def __get_MNIST_test_dataloader(self):
        # Method that loads in MNIST testing data and returns a dataloader

        test_dataset = torchvision.datasets.MNIST('.', train=False, download=True, transform=self.transform)

        # Filter out samples corresponding to classes other than 4 and 9
        mask = (test_dataset.targets == 4) | (test_dataset.targets == 9)
        test_dataset = torch.utils.data.Subset(test_dataset, np.where(mask)[0])
        test_dataset.dataset.targets = test_dataset.dataset.targets == 9

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size_test, shuffle=self.shuffle
        )

        return test_loader

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

        curr_dist.append(len(ds.targets[ds.targets == 4]))
        curr_dist.append(len(ds.targets[ds.targets == 9]))

        return curr_dist

    def sample_bias(self, desired_sample_distribution, dataset="train", dist_is_freq=True):
        # Force a sample bias on the specified dataset. Note that samples will be removed in order
        # to enforce the provided distribution. The maximum number of samples will be retained, however.
        # dist_is_freq: True if provided distribution is a list of frequencies. False if it's a list of absolute numbers desired.

        ds = self.__get_dataset(dataset)

        class_labels = [4, 9]

        if dist_is_freq:
            dist = np.array(desired_sample_distribution)
            dist_freq = dist / np.sum(dist)
            curr_dist = np.array(self.get_curr_dist(dataset=dataset))
            max_num = int(np.min(curr_dist / dist_freq))
            desired_sample_distribution = np.floor(max_num * dist_freq).astype(int)

        sample_index = np.empty((0,), dtype=int)
        for label, n in zip(class_labels, desired_sample_distribution):
            curr_idx = np.where(ds.targets == label)[0]
            sampled_indices = np.random.choice(curr_idx, size=n, replace=False)
            sample_index = np.concatenate((sample_index, sampled_indices))

        sample_index = np.sort(sample_index)
        ds.targets = ds.targets[sample_index] == 9
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
