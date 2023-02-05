# %% [markdown]
# # Import required packages

# %%
## Standard libraries
import os
import numpy as np
import random
import json
from PIL import Image
from collections import defaultdict
from statistics import mean, stdev
from copy import deepcopy
import pickle
import re

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
# %matplotlib inline
from IPython.display import set_matplotlib_formats
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.auto import tqdm

## PyTorch
import torch
torch.manual_seed(0) # Set seed for pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR100, SVHN
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
pl.seed_everything(42) # Set seed for pytorch lightning

# Import tensorboard
# %load_ext tensorboard

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# %% [markdown]
# # Set folder paths

# %%
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "raw_data/seed-v/merged_data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models/seed-v"

# %% [markdown]
# # Define dataset class for SEEDV and initialize a new dataset

# %%
class SEEDV(data.Dataset):
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
        self.tensor_dataset = self.load_data()

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
        tensor_dataset = data.TensorDataset(torch.from_numpy(
            subjects), torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1]))

        return tensor_dataset

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
        all_subjects = None

        for s_num in s_nums:
            if all_features == None:
                all_features, all_emotions, all_subjects = self.__getitem__(s_num)
            else:
                features, targets, subjects = self.__getitem__(s_num)
                all_features = torch.vstack((all_features, features))
                all_emotions = torch.vstack((all_emotions, targets))
                all_subjects = torch.vstack((all_subjects, subjects))

        return all_features, all_emotions.reshape((1,-1))[0], all_subjects.reshape((1,-1))[0]

    def __len__(self):
        """get number of samples in dataset

        Returns:
            int: number of samples in dataset
        """
        return len(self.tensor_dataset)

    def __getitem__(self, s_num: int):
        """get data for a given s_num

        Args:
            s_num (int): number that identifies a subject

        Returns:
            tuple of torch.tensor: features and corresponding emotion numbers
        """
        indices = torch.where(self.tensor_dataset[:][0] == s_num, True, False)
        # all of the following has the length of 1823
        # features = tensor([[11.0517, 9.1559, 9.5510, ...]])
        features = self.tensor_dataset[indices][1]
        # emotion_nums = tensor([4, 4, 4,  ..., 0, 0, 0])
        emotion_nums = self.tensor_dataset[indices][2]
        # s_nums = tensor([6, 6, 6,  ..., 6, 6, 6]) always s_num
        s_nums = torch.full((len(emotion_nums),), s_num, dtype=torch.int64)
        
        return features, emotion_nums.to(torch.int64).reshape((1,-1))[0], s_nums.reshape((1,-1))[0]

# %%
seed_v = SEEDV(emotion_dict = {0: 'disgust', 1: 'fear', 2: 'sad', 3: 'neutral', 4: 'happy'}, num_subjects=16, data_path=DATASET_PATH)

# %% [markdown]
# # Perform a train-val-test split by label

# %%
tasks = torch.randperm(16) # random permutation of subjects 0 to 15 which are the metaclasses
train_tasks, val_tasks, test_tasks = tasks[0:12], tasks[12:14], tasks[14:16]

# %%
class EEGDataset(data.Dataset):
    def __init__(self, features, targets, subjects):
        super().__init__()
        self.features = features
        self.targets = targets
        self.subjects = subjects

    def __getitem__(self, idx):
        feature, target, subject = self.features[idx], self.targets[idx], self.subjects[idx]

        return feature, target, subject

    def __len__(self):
        return self.features.shape[0]

# %%
# this dataset contains data from subjects 0 - 11 (training subjects)
train_set = EEGDataset(*seed_v.get_subjects_data(train_tasks))

# this dataset contains data from validation subjects
val_set = EEGDataset(*seed_v.get_subjects_data(val_tasks))

# this dataset contains data from testing subjects
test_set = EEGDataset(*seed_v.get_subjects_data(test_tasks))

# %% [markdown]
# # Setup dataloaders and samplers

# %%
class FewShotBatchSampler(object):

    def __init__(self, dataset_targets, dataset_subjects, N_way, K_shot, include_query=False, shuffle=False, shuffle_once=False):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which 
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but 
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in 
                           the beginning, but kept constant across iterations 
                           (for validation)
        """
        super().__init__()
        self.dataset_targets = dataset_targets # emotion labels
        self.dataset_subjects = dataset_subjects # subject labels
        # dataset_targets[n] is associated with dataset_subjects[n]? yes     
        
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot
    
        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        
        ###### TESTING BEGIN ######
        self.subjects = torch.unique(self.dataset_subjects).tolist()
        self.num_subjects = len(self.subjects)
        self.indices_per_subject = {}
        self.samples_per_subject = {}
        self.start_index = 0
        self.current_subject = 0
        self.visited = {}
    
        for s in self.subjects:
            self.visited[s] = []
            self.indices_per_subject[s] = torch.where(self.dataset_subjects == s)[0]
            self.samples_per_subject[s] = self.indices_per_subject[s].shape[0]
        ###### TESTING END ######

    ###### TESTING BEGIN ######
    def select_subject_batch(self, subject_num):
        selected_count = defaultdict(int)
        for i in range(self.num_classes):
            selected_count[i] = 0
        
        selected_indices = []
        for i in range(self.start_index, self.start_index+self.samples_per_subject[subject_num]):
            if i not in self.visited[subject_num]:
                cur = self.dataset_targets[i].item()
                if selected_count[cur] < self.K_shot:
                    selected_indices.append(i)
                    selected_count[cur] += 1
                    self.visited[subject_num].append(i)
                elif sum(selected_count.values()) == self.batch_size:
                    break
        while sum(selected_count.values()) < self.batch_size:
            for i in range(self.start_index, self.start_index+self.samples_per_subject[subject_num]):
                # if i not in self.visited:
                cur = self.dataset_targets[i].item()
                if selected_count[cur] < self.K_shot:
                    selected_indices.append(i)
                    selected_count[cur] += 1
                    self.visited[subject_num].append(i)
                elif sum(selected_count.values()) == self.batch_size:
                    break
            # handle all idx is being marked as visited
            if len(self.visited[subject_num]) == self.samples_per_subject[subject_num]:
                # all marked! cant select anymore
                break

        return selected_indices
    ###### TESTING END ######

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for 
        # training and validation, this is not a problem.
        # random.shuffle(self.class_list)
        ###### TESTING BEGIN ######
        ###### TESTING END ######

        

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # # Sample few-shot batches
        # start_index = defaultdict(int)
        # for it in range(self.iterations):
        #     # Select N classes for the batch
        #     # class_batch = self.class_list[it*self.N_way:(it+1)*self.N_way]
        #     class_batch = random.sample(self.classes, self.N_way)
        #     index_batch = []
        #     for c in class_batch:  # For each class, select the next K examples and add them to the batch
        #         index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c]+self.K_shot])
        #         start_index[c] += self.K_shot
        #         try:
        #             self.indices_per_class[c][start_index[c]]
        #         except:
        #             start_index[c] = 0
        #     if self.include_query:  # If we return support+query set, sort them so that they are easy to split
        #         index_batch = index_batch[::2] + index_batch[1::2]
        #     yield index_batch
        
        ###### TESTING BEGIN ######
        # keep track of which subject we are on.
        # self.current_subject = 0
        # we need to move on to the next subject when num_marked % num_samples_per_subject == 0
        for s in self.subjects:
            # select the subject batch
            while (len(self.visited[s]) < self.samples_per_subject[s]):
                # select the subject batch
                selected_indices = self.select_subject_batch(s)
                # shuffle the indices
                selected_indices = random.sample(selected_indices, len(selected_indices))
                # yield the selected indices
                yield selected_indices
            
            self.start_index += self.samples_per_subject[s]
        
        ###### TESTING END ######

    def __len__(self):
        return self.iterations

# %%
N_WAY = 5
K_SHOT = 10
train_data_loader = data.DataLoader(train_set,
                                    batch_sampler=FewShotBatchSampler(train_set.targets,
                                                                      train_set.subjects,
                                                                      include_query=True,
                                                                      N_way=N_WAY,
                                                                      K_shot=K_SHOT,
                                                                      shuffle=True),
                                    num_workers=32)
val_data_loader = data.DataLoader(val_set,
                                  batch_sampler=FewShotBatchSampler(val_set.targets,
                                                                    val_set.subjects,
                                                                    include_query=True,
                                                                    N_way=N_WAY,
                                                                    K_shot=K_SHOT,
                                                                    shuffle=False,
                                                                    shuffle_once=True),
                                  num_workers=32)

# %% [markdown]
# # Split batch into query and support

# %%
def split_query_support(features, targets):
    support_features, query_features = features.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_features, query_features, support_targets, query_targets

# %%
features, targets, _ = next(iter(val_data_loader))
print(features)
input()
print(targets)
input()
support_features, query_features, support_targets, query_targets = split_query_support(features, targets)

# fig, ax = plt.subplots(1, 2, figsize=(8, 5))
# ax[0].plot(support_features, 'o')
# ax[0].set_title("Support set")
# ax[0].axis('off')
# ax[1].plot(query_features, 'o')
# ax[1].set_title("Query set")
# ax[1].axis('off')
# plt.suptitle("Few Shot Batch", weight='bold')
# plt.show()
# plt.close()

# %% [markdown]
# # Normalize and visualize the features in batch

# %%
# Normalize the features
support_features = support_features / support_features.max(0, keepdim=True)[0]
query_features = query_features / query_features.max(0, keepdim=True)[0]

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
support_pca = pca.fit_transform(support_features)
query_pca = pca.fit_transform(query_features)

fig, ax = plt.subplots(1, 2, figsize=(8, 5))
ax[0].scatter(support_pca[:,0], support_pca[:,1], c=support_targets)
ax[0].set_title("Support set")
ax[0].axis('off')
ax[1].scatter(query_pca[:,0], query_pca[:,1], c=query_targets)
ax[1].set_title("Query set")
ax[1].axis('off')
plt.suptitle("Few Shot Batch", weight='bold')
plt.show()
plt.close()

# %% [markdown]
# # Define DCCA neural network used in SEEDV paper

# %% [markdown]
# ## Define CCA methods and classes

# %%
def cca_metric_derivative(H1, H2):
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9
    # transform the matrix: to be consistent with the original paper
    H1 = H1.T
    H2 = H2.T
    # o1 and o2 are feature dimensions
    # m is sample number
    o1 = o2 = H1.shape[0]
    m = H1.shape[1]

    # calculate parameters
    H1bar = H1 - H1.mean(axis=1).reshape([-1,1])
    H2bar = H2 - H2.mean(axis=1).reshape([-1,1])

    SigmaHat12 = (1.0 / (m - 1)) * np.matmul(H1bar, H2bar.T)
    SigmaHat11 = (1.0 / (m - 1)) * np.matmul(H1bar, H1bar.T) + r1 * np.eye(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.matmul(H2bar, H2bar.T) + r2 * np.eye(o2)

    # eigenvalue and eigenvector decomposition
    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)

    # remove eighvalues and eigenvectors smaller than 0
    posInd1 = np.where(D1 > 0)[0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]

    posInd2 = np.where(D2 > 0)[0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    # calculate matrxi T
    SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)
    Tval = np.matmul(np.matmul(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)
    # By default, we will use all the singular values
    tmp = np.matmul(Tval.T, Tval)
    corr = np.sqrt(np.trace(tmp))
    cca_loss = -1 * corr

    # calculate the derivative of H1 and H2
    U_t, D_t, V_prime_t = np.linalg.svd(Tval)
    Delta12 = SigmaHat11RootInv @ U_t @ V_prime_t @ SigmaHat22RootInv
    Delta11 = SigmaHat11RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat11RootInv
    Delta22 = SigmaHat22RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat22RootInv
    Delta11 = -0.5 * Delta11
    Delta22 = -0.5 * Delta22

    DerivativeH1 = ( 1.0 / (m - 1)) * (2 * (Delta11 @ H1bar) + Delta12 @ H2bar)
    DerivativeH2 = ( 1.0 / (m - 1)) * (2 * (Delta22 @ H2bar) + Delta12 @ H1bar)

    return cca_loss, DerivativeH1.T, DerivativeH2.T

class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        breakpoint()
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

# %% [markdown]
# ## Define DCCA network layers

# %%
class TransformLayers(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(TransformLayers, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    nn.Sigmoid(),
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id+1], affine=False),
                    ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, output_dim):
        super(AttentionFusion, self).__init__()
        self.output_dim = output_dim
        self.attention_weights = nn.Parameter(torch.randn(self.output_dim, requires_grad=True))
    def forward(self, x1, x2):
        # calculate weigths for all input samples
        row, _ = x1.shape
        fused_tensor = torch.empty_like(x1)
        alpha = []
        for i in range(row):
            tmp1 = torch.dot(x1[i,:], self.attention_weights)
            tmp2 = torch.dot(x2[i,:], self.attention_weights)
            alpha_1 = torch.exp(tmp1) / (torch.exp(tmp1) + torch.exp(tmp2))
            alpha_2 = 1 - alpha_1
            alpha.append((alpha_1.detach().cpu().numpy(), alpha_2.detach().cpu().numpy()))
            fused_tensor[i, :] = alpha_1 * x1[i,:] + alpha_2 * x2[i, :]
        return fused_tensor, alpha

class DCCA_AM(nn.Module):
    def __init__(self, input_size1, input_size2, layer_sizes1, layer_sizes2, outdim_size, categories, device):
        super(DCCA_AM, self).__init__()
        self.input_dim_split = input_size1
        self.outdim_size = outdim_size
        self.categories = categories
        # self.use_all_singular_values = use_all_singular_values
        self.device = device

        self.model1 = TransformLayers(input_size1, layer_sizes1).to(self.device)
        self.model2 = TransformLayers(input_size2, layer_sizes2).to(self.device)

        # convert generator object to list for deepcopy(model) to work
        self.model1_parameters = list(self.model1.parameters()) 
        self.model2_parameters = list(self.model2.parameters())

        self.classification = nn.Linear(self.outdim_size, self.categories)

        self.attention_fusion = AttentionFusion(outdim_size)
    def forward(self, x):
        x1 = x[:, :self.input_dim_split]
        x2 = x[:, self.input_dim_split:]
        # forward process: returns negative of cca loss and predicted labels
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        # cca_loss_val = self.loss(output1, output2)
        cca_loss, partial_h1, partial_h2 = cca_metric_derivative(output1.detach().cpu().numpy(), output2.detach().cpu().numpy())
        fused_tensor, alpha = self.attention_fusion(output1, output2)
        out = self.classification(fused_tensor)
        return out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor.detach().cpu().data, alpha

# %% [markdown]
# # Define meta learning model and methods

# %% [markdown]
# ## Define baseline ProtoNet model

# %%
class ProtoNet(pl.LightningModule):

    def __init__(self, lr, model_args):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        eeg_input_dim, eye_input_dim, layer_sizes1, layer_sizes2, \
            output_dim, num_emotions, device = model_args
        self.model = DCCA_AM(eeg_input_dim, eye_input_dim, layer_sizes1, layer_sizes2, output_dim, num_emotions, device).to(device)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        features = features[0]
        # targets = targets.reshape((1,-1))[0] 
        # already RESHAPED EARLIER in dataset_from_labels call
        classes, _ = torch.unique(targets).sort() # Determine which classes we have
        prototypes = []
        # print("targets:", targets)
        for c in classes:
            # print("c:", c)
            # print(features[torch.where(targets == c)[0]])
            # maybe use for target in targets loop
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    def classify_feats(self, prototypes, classes, feats, targets):
        # Classify new examples with prototypes and return classification error
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc

    def calculate_loss(self, batch, mode):
        # Determine training loss for a given support and query set 
        features, targets = batch
        outputs = self.model(features)  # Encode all images of support and query set
        support_feats, query_feats, support_targets, query_targets = split_query_support(outputs, targets)
        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)
        preds, labels, acc = self.classify_feats(prototypes, classes, query_feats, query_targets)
        loss = F.cross_entropy(preds, labels)

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self.calculate_loss(batch, mode="val")

# %% [markdown]
# ## Define ProtoMAML model

# %%
class ProtoMAML(pl.LightningModule):
    
    def __init__(self, lr, lr_inner, lr_output, num_inner_steps, model_args):
        """
        Inputs
            eeg_input_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        super().__init__()
        self.save_hyperparameters()
        eeg_input_dim, eye_input_dim, layer_sizes1, layer_sizes2, \
            output_dim, num_emotions, device = model_args
        self.model = DCCA_AM(eeg_input_dim, eye_input_dim, layer_sizes1, layer_sizes2, output_dim, num_emotions, device).to(device)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140,180], gamma=0.1)
        return [optimizer], [scheduler]
        
    def run_model(self, local_model, output_weight, output_bias, features, target_indices):
        # Execute a model with given output layer weights and inputs
        dcca_out = local_model(features)
        out = dcca_out[0]
        cca_loss = dcca_out[1]
        preds = F.linear(out, output_weight, output_bias)
        # loss = F.cross_entropy(preds, labels.reshape((-1,1))[0]) 
        # already RESHAPED EARLIER in dataset_from_labels calls
        loss = 0.7 * cca_loss + 1.0 * F.cross_entropy(preds, target_indices)
        acc = (preds.argmax(dim=1) == target_indices).float()
        return loss, preds, acc
        
    def adapt_few_shot(self, support_features, support_targets):
        # Determine prototype initialization
        support_feats = self.model(support_features)
        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)
        support_labels = (classes[None,:] == support_targets[:,None]).long().argmax(dim=-1)
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()
        
        # Optimize inner loop model on support set
        for _ in range(self.hparams.num_inner_steps):
            # Determine loss on the support set
            loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_features, support_labels)
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            output_weight.data -= self.hparams.lr_output * output_weight.grad
            output_bias.data -= self.hparams.lr_output * output_bias.grad
            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)
            
        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias
        
        return local_model, output_weight, output_bias, classes
        
    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []
        self.model.zero_grad()
        
        # Determine gradients for batch of tasks
        for task_batch in batch:
            features, targets = task_batch
            support_features, query_features, support_targets, query_targets = split_query_support(features, targets)
            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_features, support_targets)
            # Determine loss of query set
            query_labels = (classes[None,:] == query_targets[:,None]).long().argmax(dim=-1)
            loss, preds, acc = self.run_model(local_model, output_weight, output_bias, query_features, query_labels)
            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model
            
            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())
        
        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()
        
        self.log(f"{mode}_loss", sum(losses) / len(losses))
        self.log(f"{mode}_acc", sum(accuracies) / len(accuracies))
    
    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")
        return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning
        
    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)
    
    def test_step(self, batch, batch_idx):
        # Testing requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="test")
        torch.set_grad_enabled(False)

# %% [markdown]
# ## Define task batch sampler

# %%
class TaskBatchSampler(object):
    
    def __init__(self, dataset_targets, dataset_subjects, batch_size, N_way, K_shot, include_query=False, shuffle=True):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            batch_size - Number of tasks to aggregate in a batch
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which 
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but 
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
        """
        super().__init__()
        # TODO: update param to pass in dataset_subjects
        self.batch_sampler = FewShotBatchSampler(dataset_targets, dataset_subjects, N_way, K_shot, include_query, shuffle)
        self.task_batch_size = batch_size
        self.local_batch_size = self.batch_sampler.batch_size
        
    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            batch_list.extend(batch)
            if (batch_idx+1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []
        
    def __len__(self):
        return len(self.batch_sampler)//self.task_batch_size
    
    def get_collate_fn(self):
        # Returns a collate function that converts one big tensor into a list of task-specific tensors
        def collate_fn(item_list):
            features = torch.stack([feat for feat, target in item_list], dim=0)
            targets = torch.stack([target for feat, target in item_list], dim=0)
            features = features.chunk(self.task_batch_size, dim=0)
            targets = targets.chunk(self.task_batch_size, dim=0)
            return list(zip(features, targets))
        return collate_fn

# %% [markdown]
# # Train meta learning model

# %%
def train_model(model_class, train_loader, val_loader, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=30,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", every_n_epochs=1),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         log_every_n_steps=10)
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, model_class.__name__ + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = model_class(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    return model, trainer

# %%
# Training constant (same as for ProtoNet)
N_WAY = 5
K_SHOT = 10

# TODO: implement 1 dataloader for each subject's data to act as the tasks

# Training set
train_protomaml_sampler = TaskBatchSampler(train_set.targets,
                                           train_set.subjects, 
                                           include_query=True,
                                           N_way=N_WAY,
                                           K_shot=K_SHOT,
                                           batch_size=len(train_set.targets) // len(train_tasks))

train_protomaml_loader = data.DataLoader(train_set, 
                                         batch_sampler=train_protomaml_sampler,
                                         collate_fn=train_protomaml_sampler.get_collate_fn(),
                                         num_workers=32)

# Validation set
val_protomaml_sampler = TaskBatchSampler(val_set.targets, 
                                         val_set.subjects,
                                         include_query=True,
                                         N_way=N_WAY,
                                         K_shot=K_SHOT,
                                         batch_size=len(val_set.targets) // len(val_tasks),  # We do not update the parameters, hence the batch size is irrelevant here
                                         shuffle=False)
val_protomaml_loader = data.DataLoader(val_set, 
                                       batch_sampler=val_protomaml_sampler,
                                       collate_fn=val_protomaml_sampler.get_collate_fn(),
                                       num_workers=32)

# %%
EEG_INPUT_DIM = 310
EYE_INPUT_DIM = 33
OUTPUT_DIM = 12
LAYER_SIZES = [200, 50, OUTPUT_DIM]
NUM_EMOTIONS = N_WAY

protomaml_model, trainer = train_model(ProtoMAML, 
                              lr=1e-3, 
                              lr_inner=0.1,
                              lr_output=0.1,
                              num_inner_steps=1,
                              model_args = (EEG_INPUT_DIM, EYE_INPUT_DIM, LAYER_SIZES,
                                LAYER_SIZES, OUTPUT_DIM, NUM_EMOTIONS, device),
                              train_loader=train_protomaml_loader, 
                              val_loader=val_protomaml_loader)

# %% [markdown]
# # Test meta learning model

# %%
def test_protomaml(model, trainer, dataset, n_way, k_shot=4):
    pl.seed_everything(42)
    model = model.to(device)
    num_classes = dataset.targets.unique().shape[0]

    # Data loader for full test set as query set
    full_dataloader = data.DataLoader(dataset, batch_size=1, num_workers=32, shuffle=False, drop_last=False)
    # Data loader for sampling support sets
    sampler = FewShotBatchSampler(
        dataset.targets, dataset.subjects, include_query=False, N_way=n_way, K_shot=k_shot, shuffle=False, shuffle_once=False
    )
    sample_dataloader = data.DataLoader(dataset, batch_sampler=sampler, num_workers=32)

    # We iterate through the full dataset in two manners. First, to select the k-shot batch.
    # Second, the evaluate the model on all other examples
    accuracies = []
    for (support_imgs, support_targets), support_indices in tqdm(
        zip(sample_dataloader, sampler), "Performing few-shot finetuning"
    ):
        support_imgs = support_imgs.to(device)
        support_targets = support_targets.to(device)
        # Finetune new model on support set
        local_model, output_weight, output_bias, classes = model.adapt_few_shot(support_imgs, support_targets)
        with torch.no_grad():  # No gradients for query set needed
            local_model.eval()
            batch_acc = torch.zeros((0,), dtype=torch.float32, device=device)
            # Evaluate all examples in test dataset
            for query_imgs, query_targets in full_dataloader:
                query_imgs = query_imgs.to(device)
                query_targets = query_targets.to(device)
                query_labels = (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
                _, _, acc = model.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)
                batch_acc = torch.cat([batch_acc, acc.detach()], dim=0)
            # Exclude support set elements
            for s_idx in support_indices:
                batch_acc[s_idx] = 0
            batch_acc = batch_acc.sum().item() / (batch_acc.shape[0] - len(support_indices))
            accuracies.append(batch_acc)
    return mean(accuracies), stdev(accuracies)

# %%
# Reduce inner learning rate for few shot and increase number of inner steps, if necessary, for further 
# improvement during testing phase
protomaml_model.hparams.lr_inner = 0.1
protomaml_model.hparams.lr_output = 0.1
protomaml_model.hparams.num_inner_steps = 10

# %%
protomaml_result_file = os.path.join(CHECKPOINT_PATH, "protomaml_fewshot.json")

# Perform same experiments as for ProtoNet
protomaml_accuracies = dict()
for k in [5, 10, 15]:
    protomaml_accuracies[k] = test_protomaml(protomaml_model, trainer, test_set, n_way=N_WAY, k_shot=k)
    print(f"Accuracy for k={k}: {100.0*protomaml_accuracies[k][0]:4.2f}% (+-{100.0*protomaml_accuracies[k][1]:4.2f}%)")
# Export results
with open(protomaml_result_file, 'w') as f:
    json.dump(protomaml_accuracies, f, indent=4)

# %%
def plot_few_shot(acc_dict, name, color=None, ax=None):
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,3))
    ks = sorted(list(acc_dict.keys()))
    mean_accs = [acc_dict[k][0] for k in ks]
    std_accs = [acc_dict[k][1] for k in ks]
    ax.plot(ks, mean_accs, marker='o', markeredgecolor='k', markersize=6, label=name, color=color)
    ax.fill_between(ks, [m-s for m,s in zip(mean_accs, std_accs)], [m+s for m,s in zip(mean_accs, std_accs)], alpha=0.2, color=color)
    ax.set_xticks(ks)
    ax.set_xlim([ks[0]-1, ks[-1]+1])
    ax.set_xlabel("Number of shots per class", weight='bold')
    ax.set_ylabel("Accuracy", weight='bold')
    if len(ax.get_title()) == 0:
        ax.set_title("Few-Shot Performance " + name, weight='bold')
    else:
        ax.set_title(ax.get_title() + " and " + name, weight='bold')
    ax.legend()
    return ax

# %%
plot_few_shot(protomaml_accuracies, name="ProtoMAML", color="C2", ax=None)
plt.show()
plt.close()


