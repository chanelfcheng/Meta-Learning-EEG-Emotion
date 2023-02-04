import os
import re
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "raw_data/seed-v/merged_data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models/seed-v"

N_WAY = 5 # number of subjects
K_SHOT = 12 # number of samples to learn from per subject

# %% [markdown]
# # Define dataset class for SEEDV and initialize a new dataset

# %%
class SEEDV():
    """SEEDV dataset"""
    def __init__(self, emotion_dict: dict, num_subjects: int, data_path: str):
        """constructor for SEEDV dataset

        Args:
            emotion_dict (dict): dictionary of emotion_num as key and
            emotion_str as value
            num_subjects (int): number of subjects in the dataset
            data_path (str): path to the directory containing npy data files
        """
        self.emotion_dict = emotion_dict
        self.num_subjects = num_subjects
        self.data_path = data_path
        self.dataset = self.load_data()

    def load_data(self):
        """load data from npy files to create tensor dataset

        Returns:
            data.TensorDataset: tensor dataset containing 
                all data from all npy files
        """
        # initialize dataset and subjects to None at start
        dataset = None
        subjects = None

        # loop through all files in the directory
        for file in os.listdir(self.data_path):
            # get the file path of the file
            file_path = os.path.join(self.data_path, file)

            # get subject number from the filename
            s_num = np.int64(re.findall("\d+", file)[0]) - 1

            # if dataset has not been created yet
            if dataset is None:
                # create subject meta labels for each sample in dataset
                for i in range(len(np.load(file_path))):
                    if subjects is None:
                        subjects = s_num
                    else:
                        # need to hstack subjects to match tensor shapes of emotion
                        subjects = np.hstack((subjects, s_num))
                # load the data from the npy file
                dataset = np.load(file_path)
            # else if dataset already exists
            else:
                for i in range(len(np.load(file_path))):
                    # hstack subjects to match tensor shapes of emotion
                    subjects = np.hstack((subjects, s_num))
                # stack the data vertically
                dataset = np.vstack((dataset, np.load(file_path)))

        # create tensor dataset with subject meta labels, features, and emotion labels
        dataset = (subjects, dataset[:, :-1], dataset[:, -1])

        return dataset

    def get_subjects_data(self, s_nums: list):
        """get data for only subjects in s_nums

        Args:
            s_nums (list): list of subject numbers that identifies subjects

        Returns:
            tuple of torch.tensor: all features and all emotion numbers for only
            specified subjects
        """
        all_features = None
        all_emotions = None

        for s_num in s_nums:
            if type(all_features) == type(None):
                all_features, all_emotions = self.__getitem__(s_num)
            else:
                features, targets = self.__getitem__(s_num)
                all_features = np.vstack((all_features, features))
                all_emotions = np.vstack((all_emotions, targets))

        return all_features, all_emotions.reshape((1,-1))[0]

    def __len__(self):
        """get number of samples in dataset

        Returns:
            int: number of samples in dataset
        """
        return len(self.dataset)

    def __getitem__(self, s_num: int):
        """get data for a given s_num

        Args:
            s_num (int): number that identifies a subject

        Returns:
            tuple of torch.tensor: features and corresponding emotion numbers
        """
        indices = np.where(self.dataset[0][:] == s_num, True, False)
        features = self.dataset[1][indices]
        emotion_num = self.dataset[2][indices]

        return features, emotion_num.astype(np.int64).reshape((1,-1))[0]

# %%
emotion_dict = {0: 'disgust', 1: 'fear', 2: 'sad', 3: 'neutral', 4: 'happy'}
seed_v = SEEDV(emotion_dict=emotion_dict, num_subjects=16, data_path=DATASET_PATH)

# %% [markdown]
# # Perform a train-val-test split by label

# %%
tasks = np.random.permutation(16) # random permutation of subjects 0 to 15 which are the metaclasses
train_tasks, val_tasks, test_tasks = tasks[0:12], tasks[12:14], tasks[14:16]

# %%
class EEGDataset():
    """EEG Dataset for one subject"""
    def __init__(self, features, targets):
        super().__init__()
        self.features = features
        self.targets = targets

    def __getitem__(self, idx):
        features, targets = self.features[idx], self.targets[idx]

        return features, targets

    def __len__(self):
        return self.features.shape[0]

import torch
from torch import nn
class StreamingLDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant Analysis algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, test_batch_size=1024, shrinkage_param=1e-4,
                 streaming_update_sigma=True):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param test_batch_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        """

        super(StreamingLDA, self).__init__()

        # SLDA parameters
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():

            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

            # update class means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1

    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                        self.device))
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Sigma'] = self.Sigma.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_path, save_name):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_path, save_name + '.pth'))
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Sigma = d['Sigma'].to(self.device)
        self.num_updates = d['num_updates']

features_train = seed_v.get_subjects_data(train_tasks)[0]
targets_train = seed_v.get_subjects_data(train_tasks)[1]

model = make_pipeline(MinMaxScaler(), StreamingLDA(input_shape=features_train.shape[1], num_classes=len(emotion_dict)), SVC(kernel='rbf', gamma='scale'))

model.fit(features_train, targets_train)

features_test = seed_v.get_subjects_data(test_tasks)[0]
targets_test = seed_v.get_subjects_data(test_tasks)[1]
results = model.predict(features_test)

from sklearn.metrics import classification_report
print(classification_report(targets_test, results, target_names=emotion_dict.values()))
# %%
