# %% [markdown]
# # Import required packages

# %%
## Standard libraries
import sys
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
pl.seed_everything(42)

# Import tensorboard
# %load_ext tensorboard

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
    def __init__(self, emotion_dict, num_participants, data_dir):
        self.emotion_dict = emotion_dict
        self.num_participants = num_participants
        self.data_dir = data_dir
        self.tensor_dataset = self.load_data()
    
    def load_data(self):
        dataset = None           
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if dataset is None:
                target = np.int64(re.findall("\d+", file)[0])
                for i in range(1, len(np.load(file_path))):
                    target = np.hstack((target, int(re.findall("\d+", file)[0])))
                dataset = np.load(file_path)
            else:
                for i in range(len(np.load(file_path))):
                    target = np.hstack((target, np.int64(re.findall("\d+", file)[0])))
                dataset = np.vstack((dataset, np.load(file_path)))

        tensor_dataset = data.TensorDataset(torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1]), torch.from_numpy(target))
                    
        return tensor_dataset
    
    def get_all_data(self):
        """
        same as self.__getitem__(self, idx) but instead of a specific index
        this function will return all the tuple that contains all features and 
        combined labels 
        """
        all_features = None
        all_combined_labels = None

        for idx in range(len(self.tensor_dataset)):
            if all_features == None:
                all_features, all_combined_labels = self.__getitem__(idx)
            else:
                features, combined_labels = self.__getitem__(idx)
                all_features = torch.vstack((all_features, features))
                all_combined_labels = torch.vstack((all_combined_labels, combined_labels))
        
        return all_features, all_combined_labels

    def __len__(self):
        return len(self.tensor_dataset)
        
    def __getitem__(self, idx):
        """
        return a tuple of features, combined_label (from participant and emotion)
        p1: 0 1 2 3 4,
        p2: 5 6 7 8 9,
        ...
        p16: 75 76 77 78 79

        we are only given participant # and emotion #
        p1_0: 0 = (1 - 1) * 5
        p1_1: 1 = 0 + 1
        p1_2: 2 = 0 + 2
        p1_3: 3 = 0 + 3
        p1_4: 4 = 0 + 4
        p2_0: 5 = (2 - 1) * 5
        p2_1: 6 = 5 + 1
        p2_2: 7 = 5 + 2
        p2_3: 8 = 5 + 3
        p2_4: 9 = 5 + 4
        ..
        p16_0: 75 = (16 - 1) * 5 = 75
        ...
        """
        features = self.tensor_dataset[idx][0]
        emotion_num = self.tensor_dataset[idx][1]
        participant_num = self.tensor_dataset[idx][2]
        base = (participant_num - 1) * len(self.emotion_dict)
        combined_label = base + emotion_num

        return features, combined_label.to(torch.int64)

# %%
seed_v = SEEDV(emotion_dict = {0: 'disgust', 1: 'fear', 2: 'sad', 3: 'neutral', 4: 'happy'}, num_participants=16, data_dir=DATASET_PATH)

# %% [markdown]
# # Perform a train-val-test split by label

# %%
classes = torch.randperm(5*16) # Generate random permutation of numbers from 0 to 79
train_classes, val_classes, test_classes = classes[:64], classes[64:72], classes[72:] # 80-10-10 split

# %%
SEEDV_all_features, SEEDV_all_labels = seed_v.get_all_data()

# %%
class EEGDataset(data.Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        features, labels = self.features[idx], self.labels[idx]

        return features, labels

    def __len__(self):
        return self.features.shape[0]

# %%
def dataset_from_labels(features, labels, class_set): 
    # for label in labels:
    #     print(label)
    class_mask = (labels[:,None] == class_set[None,:]).any(dim=-1) # reshape class mask [[64], [64],... ] -> [64, 64, ...]
    return EEGDataset(features[class_mask], labels[class_mask]) # reshape labels [[0], [1], ...] -> [0, 1, ...]

# %%
train_set = dataset_from_labels(
    SEEDV_all_features, SEEDV_all_labels.reshape((1,-1))[0], train_classes)
val_set = dataset_from_labels(
    SEEDV_all_features, SEEDV_all_labels.reshape((1,-1))[0], val_classes)
test_set = dataset_from_labels(
    SEEDV_all_features, SEEDV_all_labels.reshape((1,-1))[0], test_classes)

# %% [markdown]
# # Setup dataloaders and samplers

# %%
class FewShotBatchSampler(object):

    def __init__(self, dataset_targets, N_way, K_shot, include_query=False, shuffle=True, shuffle_once=False):
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
        self.dataset_targets = dataset_targets
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
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i+p*self.num_classes for i,
                         c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for 
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it*self.N_way:(it+1)*self.N_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c]+self.K_shot])
                start_index[c] += self.K_shot
            if self.include_query:  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations

# %%
N_WAY = 5
K_SHOT = 4
train_data_loader = data.DataLoader(train_set,
                                    batch_sampler=FewShotBatchSampler(train_set.labels,
                                                                      include_query=True,
                                                                      N_way=N_WAY,
                                                                      K_shot=K_SHOT,
                                                                      shuffle=True),
                                    num_workers=32)
val_data_loader = data.DataLoader(val_set,
                                  batch_sampler=FewShotBatchSampler(val_set.labels,
                                                                    include_query=True,
                                                                    N_way=N_WAY,
                                                                    K_shot=K_SHOT,
                                                                    shuffle=False,
                                                                    shuffle_once=True),
                                  num_workers=32)

# %% [markdown]
# # Split batch into query and support

# %%
def split_query_support(features, labels):
    support_features, query_features = features.chunk(2, dim=0)
    support_labels, query_labels = labels.chunk(2, dim=0)
    return support_features, query_features, support_labels, query_labels

# %%
features, labels = next(iter(val_data_loader))
support_features, query_features, support_labels, query_labels = split_query_support(features, labels)

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
ax[0].scatter(support_pca[:,0], support_pca[:,1], c=support_labels)
ax[0].set_title("Support set")
ax[0].axis('off')
ax[1].scatter(query_pca[:,0], query_pca[:,1], c=query_labels)
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
    # if np.isnan(H2).all():
    #     print("H2 is nan")
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
            # prev_x = x # for debug purpose, remove later
            x = layer(x)
            # if torch.isnan(x).any():
            #     print("NaN detected in layer")
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
        imgs, targets = batch
        features = self.model(imgs)  # Encode all images of support and query set
        support_feats, query_feats, support_targets, query_targets = split_query_support(features, targets)
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
        
    def run_model(self, local_model, output_weight, output_bias, features, labels):
        # Execute a model with given output layer weights and inputs
        out = local_model(features)
        # get only first element of tuple in feats
        preds = F.linear(out[0], output_weight, output_bias)
        # loss = F.cross_entropy(preds, labels.reshape((-1,1))[0]) 
        # already RESHAPED EARLIER in dataset_from_labels calls
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc
        
    def adapt_few_shot(self, support_imgs, support_targets):
        # Determine prototype initialization
        support_feats = self.model(support_imgs)
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
            loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_imgs, support_labels)
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
            imgs, targets = task_batch
            support_imgs, query_imgs, support_targets, query_targets = split_query_support(imgs, targets)
            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_imgs, support_targets)
            # Determine loss of query set
            query_labels = (classes[None,:] == query_targets[:,None]).long().argmax(dim=-1)
            loss, preds, acc = self.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)
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

# %% [markdown]
# ## Define task batch sampler

# %%
class TaskBatchSampler(object):
    
    def __init__(self, dataset_targets, batch_size, N_way, K_shot, include_query=False, shuffle=True):
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
        self.batch_sampler = FewShotBatchSampler(dataset_targets, N_way, K_shot, include_query, shuffle)
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
            imgs = torch.stack([img for img, target in item_list], dim=0)
            targets = torch.stack([target for img, target in item_list], dim=0)
            imgs = imgs.chunk(self.task_batch_size, dim=0)
            targets = targets.chunk(self.task_batch_size, dim=0)
            return list(zip(imgs, targets))
        return collate_fn

# %% [markdown]
# ## Define model training and testing methods

# %%
def train_model(model_class, train_loader, val_loader, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=50,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", every_n_epochs=1),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         log_every_n_steps=12)
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

    return model

def test_protomaml(model, dataset, k_shot=4):
    pl.seed_everything(42)
    model = model.to(device)
    num_classes = dataset.labels.unique().shape[0]
    exmps_per_class = dataset.labels.shape[0]//num_classes
    
    # Data loader for full test set as query set
    full_dataloader = data.DataLoader(dataset, 
                                      batch_size=128, 
                                      num_workers=32, 
                                      shuffle=False, 
                                      drop_last=False)
    # Data loader for sampling support sets
    sampler = FewShotBatchSampler(dataset.labels, 
                                  include_query=False,
                                  N_way=num_classes,
                                  K_shot=k_shot,
                                  shuffle=False,
                                  shuffle_once=False)
    sample_dataloader = data.DataLoader(dataset, 
                                        batch_sampler=sampler,
                                        num_workers=32)
    
    # We iterate through the full dataset in two manners. First, to select the k-shot batch. 
    # Second, the evaluate the model on all other examples
    accuracies = []
    for (support_imgs, support_targets), support_indices in tqdm(zip(sample_dataloader, sampler), "Performing few-shot finetuning"):
        support_imgs = support_imgs.to(device)
        support_targets = support_targets.to(device)
        # Finetune new model on support set
        try:
            local_model, output_weight, output_bias, classes = model.adapt_few_shot(support_imgs, support_targets)
            with torch.no_grad():  # No gradients for query set needed
                local_model.eval()
                batch_acc = torch.zeros((0,), dtype=torch.float32, device=device)
                # Evaluate all examples in test dataset
                for query_imgs, query_targets in full_dataloader:
                    query_imgs = query_imgs.to(device)
                    query_targets = query_targets.to(device)
                    query_labels = (classes[None,:] == query_targets[:,None]).long().argmax(dim=-1)
                    _, _, acc = model.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)
                    batch_acc = torch.cat([batch_acc, acc.detach()], dim=0)
                # Exclude support set elements
                for s_idx in support_indices:
                    batch_acc[s_idx] = 0
                batch_acc = batch_acc.sum().item() / (batch_acc.shape[0] - len(support_indices))
                accuracies.append(batch_acc)
        except np.linalg.LinAlgError:
            print("NaN layer encountered")
    return mean(accuracies), stdev(accuracies)

# %% [markdown]
# # Train meta learning model

# %%
# Training constant (same as for ProtoNet)
N_WAY = 5
K_SHOT = 10

# Training set
train_protomaml_sampler = TaskBatchSampler(train_set.labels, 
                                           include_query=True,
                                           N_way=N_WAY,
                                           K_shot=K_SHOT,
                                           batch_size=16)
train_protomaml_loader = data.DataLoader(train_set, 
                                         batch_sampler=train_protomaml_sampler,
                                         collate_fn=train_protomaml_sampler.get_collate_fn(),
                                         num_workers=32)

# Validation set
val_protomaml_sampler = TaskBatchSampler(val_set.labels, 
                                         include_query=True,
                                         N_way=N_WAY,
                                         K_shot=K_SHOT,
                                         batch_size=1,  # We do not update the parameters, hence the batch size is irrelevant here
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

protomaml_model = ProtoMAML.load_from_checkpoint("saved_models/seed-v/ProtoMAML/lightning_logs/version_11/checkpoints/epoch=39-step=560.ckpt")

# %%
# Opens tensorboard in notebook. Adjust the path to your CHECKPOINT_PATH if needed
# %tensorboard --logdir saved_models/seed-v/ProtoMAML/lightning_logs/version_10

# %% [markdown]
# # Test meta learning model

# %%
protomaml_model.hparams.num_inner_steps = 200

# %%
print(test_set.labels)

# %%
protomaml_result_file = os.path.join(CHECKPOINT_PATH, "protomaml_fewshot.json")

if os.path.isfile(protomaml_result_file):
    # Load pre-computed results
    with open(protomaml_result_file, 'r') as f:
        protomaml_accuracies = json.load(f)
    protomaml_accuracies = {int(k): v for k, v in protomaml_accuracies.items()}
else:
    # Perform same experiments as for ProtoNet
    protomaml_accuracies = dict()
    for k in [2, 4, 8, 16, 32]:
        protomaml_accuracies[k] = test_protomaml(protomaml_model, test_set, k_shot=k)
    # Export results
    with open(protomaml_result_file, 'w') as f:
        json.dump(protomaml_accuracies, f, indent=4)

for k in protomaml_accuracies:
    print(f"Accuracy for k={k}: {100.0*protomaml_accuracies[k][0]:4.2f}% (+-{100.0*protomaml_accuracies[k][1]:4.2f}%)")

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


