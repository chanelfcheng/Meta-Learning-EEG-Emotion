import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate

from datasets import SEEDV

# Setup data path and device
data_path = os.path.join('raw_data', 'seed-v', 'merged_data')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Setup experiment info and parameters
emotion_dict = {0: 'disgust', 1: 'fear', 2: 'sad', 3: 'neutral', 4: 'happy'}
num_subjects = 16

# Define Network
class SNN(nn.Module):
    def __init__(self, num_inputs=343, num_hidden=300, num_outputs=5, num_steps=100):
        super().__init__()

        # network parameters
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps

        # surrogate function
        spike_grad = surrogate.fast_sigmoid()
        # global decay rate for all leaky neurons in layer 1
        beta1 = 0.9
        # independent decay rate for each leaky neuron in layer 2: [0, 1)
        beta2 = torch.rand((num_outputs), dtype = torch.float) #.to(device)

        # network layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, learn_beta=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad,learn_beta=True)

    def forward(self, x):

        # reset hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)

# Define EEG dataset
class EEGDataset(Dataset):
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

# Initialize network and load onto device
net = SNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

# Initialize dataset
seed_v = SEEDV(emotion_dict=emotion_dict, num_subjects=num_subjects, data_path=data_path)

# Initialize dataloaders
tasks = torch.randperm(16) # random permutation of subjects 0 to 15 which are the metaclasses
train_tasks, val_tasks, test_tasks = tasks[0:3], tasks[3:4], tasks[4:]

# train_set = EEGDataset(*seed_v.get_subjects_data(train_tasks))
# val_set =  EEGDataset(*seed_v.get_subjects_data(val_tasks))
# test_set =  EEGDataset(*seed_v.get_subjects_data(test_tasks))

full_feat, full_targ = seed_v[0]
train_set = EEGDataset(full_feat[0:1000], full_targ[0:1000])
val_set = EEGDataset(full_feat[1000:1100], full_targ[1000:1100])

train_loader = DataLoader(dataset=train_set, num_workers=32)
test_loader = DataLoader(dataset=val_set, num_workers=32)

# Training parameters
num_epochs = 1
num_steps = 1  # number of time steps per sample
batch_size = 128

# Training history
loss_hist = []
acc_hist = []

# Training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, _ = net(data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # print every 25 iterations
        if i % 25 == 0:
          net.eval()
          print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

          # check accuracy on a single batch
          acc = SF.accuracy_rate(spk_rec, targets)
          acc_hist.append(acc)
          print(f"Accuracy: {acc * 100:.2f}%\n")  # Print train accuracy

        # uncomment for faster termination
        if i == 150:
            break

# Display decay rate of layer 1 and layer 2
print(f"Trained decay rate of the first layer: {net.lif1.beta:.3f}\n")
print(f"Trained decay rates of the second layer: {net.lif2.beta}")

# Define testing function for testing on new inputs
def test_accuracy(data_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = net(data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

    result = acc/total
    print(f"Test set accuracy: {result*100:.3f}%")  # Print test accuracy

  return result

test_accuracy(test_loader, net, num_steps)
