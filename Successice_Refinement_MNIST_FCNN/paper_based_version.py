# Software for layered-resolution inference for a dense neural network
# Authors: Homa Esfahanizadeh
# Last Update: 11/21/2023

#####################################################################
# Necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17

#####################################################################
# Set computing device and set random seed
device = 'cpu'
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

#####################################################################
# Parameters and constants
DATA_SAVE_PATH = "data/"  # Location for saving datapoints
FIG_SAVE_PATH = "figures/" # Location for saving graphical results
MODEL_SAVE_PATH = "model/"  # Location for saving model weights

BATCH_SIZE = 100 # Batch size for training classifiers
N_CLASSIFIER_TRAINING_EPOCHS = 10 # Number of epochs for training classifiers
NUM_HIDDEN_NODES = 20 # Number of nodes at hidden layer of a network
NUM_LAYER = 4 # Including the input and output layers

SAVED_MODEL = True # A flag for activating the training or using the saved model

#####################################################################
# Dense neural networks that are used as classifiers

class DenseClassifier(nn.Module):
    def __init__(self, in_nodes, out_nodes, hidden_nodes=NUM_HIDDEN_NODES):
        super(DenseClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_nodes, hidden_nodes), 
            nn.ReLU(), 
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, out_nodes), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(device)
        return self.main(x)

    def train_classifier(self, train_loader, epochs=N_CLASSIFIER_TRAINING_EPOCHS):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.8)
        for epoch in tqdm(range(epochs)):
            self.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device).float(), y.to(device).float()
                y_hat = self(x).squeeze()
                loss = F.binary_cross_entropy(y_hat, y, reduction="sum")
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Average loss per sample
            avg_train_loss = train_loss / len(train_loader.dataset)
            print(f'====> Epoch: {epoch} Average loss: {avg_train_loss:.4f}')

    # This is the default inference function (baseline), we also have a custom one that enables the layered resolution
    def evaluate(self, test_loader):
        self.eval()
        test_data, test_labels = test_loader.dataset.data, test_loader.dataset.targets
        preds = self(test_data.to(device)).squeeze()
        y_score = preds.detach()
        return y_score

#####################################################################
# Partitioning each element of an array into multiple smaller numbers
# Example: for num = -13.625 and p_val = [4,0,-3], 
#           we have num_sign = -1.0 and num_parts = [13.0, 5.0]
#           This is because -13.625 = -1 * ( 13*(2**0) + 5*(2**-3))

def partition(num, p_val):
    num_sign = np.sign(num)
    num_abs = np.absolute(num)
    num_abs, _ = np.modf (num_abs / (2**p_val[0]))
    num_parts = []
    for i in range(1,np.size(p_val)):
        num_shifted = num_abs * (2**(p_val[i-1]-p_val[i]))
        num_parts.append(np.floor(num_shifted)*num_sign) 
        num_abs, _ = np.modf(num_shifted)
    return num_parts

#####################################################################
# Sigmoid Function
def sigmoid(x):
    y = x * -1
    y = np.exp(y) + 1
    y = 1 / y
    return y

#####################################################################
# The performance evaluation of a prediction
def utlity(y_score, y_true):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    y_pred = y_score > 0.5
    acc_th05 = sum(y_pred == y_true) / len(y_true)
    return [fpr, tpr, auc, acc_th05]

#####################################################################
# Relu(X+Y) = Relu(X) + relu_layered(X,Y)
def relu_layered(X,Y):
    output = np.zeros_like(X)
    output[ (X < 0) & ((X+Y) < 0) ] = 0
    output[ (X < 0) & ((X+Y) >= 0) ] = X[(X<0)&((X+Y)>=0)]+Y[(X < 0)&((X+Y) >= 0)]
    output[ (X >= 0) & ((X+Y) < 0) ] = X[(X>=0)&((X+Y)<0)]*-1
    output[ (X >= 0) & ((X+Y) >= 0) ] = Y[(X>=0)&((X+Y)>=0)]
    return output

#####################################################################
# Updated evaluation of a neural network, given the weight and samle changes
# The inference model is H0 -- A1 --> H1 -- relu,A2 --> H2 --> .... --> Hl ---> sigmoid --> f(X)
def evaluation_update(H, A, delta_H0, delta_A, indices, H_limit=-3):
    hidden_layer = np.concatenate((H[0][indices,:],np.ones((np.sum(indices), 1))), axis=1)
    delta_hidden_layer = np.concatenate((delta_H0[indices,:],np.zeros((np.sum(indices), 1))), axis=1)
    delta_hidden_layer = np.round(delta_hidden_layer / (2**H_limit)) * (2**H_limit)
    H[0][indices,:] = H[0][indices,:] + delta_H0[indices,:]
    for l in np.arange(0,len(A)):
        delta = np.matmul(delta_hidden_layer, np.transpose(A[l])) + np.matmul(hidden_layer+delta_hidden_layer, np.transpose(delta_A[l]))
        if ( l < (len(A) - 1) ):
            delta = relu_layered(H[l+1][indices,:],delta)
        delta = np.round(delta / (2**H_limit)) * (2**H_limit)
        hidden_layer = np.concatenate((H[l+1][indices,:],np.ones((np.sum(indices), 1))), axis=1)
        delta_hidden_layer = np.concatenate((delta,np.zeros((np.sum(indices), 1))), axis=1)
        H[l+1][indices,:] = H[l+1][indices,:] + delta
        A[l] = A[l]+delta_A[l]
    
    y_score = sigmoid(H[len(A)][indices,:]).reshape(-1)
    return y_score

####################################################################
# Arranging the Dataset: CIFAR10 Dataset
class SyntheticDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# N_TRAIN = cifar10_trainset.data.shape[0]
# N_TEST = cifar10_testset.data.shape[0]
# N_FEATURES = cifar10_trainset.data.shape[1]*cifar10_trainset.data.shape[2]*cifar10_trainset.data.shape[3] # 32x32x3
# N_OUT = 1 # Being car or not

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # original labels

# X_train = torch.Tensor(cifar10_trainset.data.reshape(-1, N_FEATURES)).float() / 256.0
# X_test = torch.Tensor(cifar10_testset.data.reshape(-1, N_FEATURES)).float() / 256.0
# y_train = np.where(torch.tensor(cifar10_trainset.targets) == 1, np.ones_like(cifar10_trainset.targets), np.zeros_like(cifar10_trainset.targets))
# y_test = np.where(torch.tensor(cifar10_testset.targets) == 1, np.ones_like(cifar10_testset.targets), np.zeros_like(cifar10_testset.targets))


mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)

N_TRAIN = mnist_trainset.data.shape[0]
N_TEST = mnist_testset.data.shape[0]
N_FEATURES = mnist_trainset.data.shape[1]*mnist_trainset.data.shape[2]
N_OUT = 1 # Being even or odd

X_train = mnist_trainset.data.reshape(-1, 784).float() / 256.0
y_train = np.where(mnist_trainset.targets % 2 == 0, np.ones(mnist_trainset.targets.shape), np.zeros_like(mnist_trainset.targets))
X_test = mnist_testset.data.reshape(-1, 784).float() / 256.0
y_test = np.where(mnist_testset.targets % 2 == 0, np.ones(mnist_testset.targets.shape), np.zeros_like(mnist_testset.targets))


train_dataset = SyntheticDataset(X_train, y_train)
test_dataset = SyntheticDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

X_test_Q = partition(X_test.numpy(), [0,-3])  # three bits seems to be sufficient for precision
X_test_Q = X_test_Q[0]
X_test_Q = X_test_Q * (2**-3)

####################################################################
# Experiments: Training classifier
model = DenseClassifier(N_FEATURES,N_OUT).to(device)
if (SAVED_MODEL == True):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH+"CIFAR10_CAR.pt"))
else:
    model.train_classifier(train_loader, epochs=N_CLASSIFIER_TRAINING_EPOCHS)
    torch.save(model.state_dict(), MODEL_SAVE_PATH+"CIFAR10_CAR.pt")

y_true = test_loader.dataset.targets

####################################################################
# Experiments on one-shot inference with full resolution pixels

y_score_oneshot = model.evaluate(test_loader).numpy()
[fpr_oneshot, tpr_oneshot, auc_oneshot, acc_th05_oneshot] = utlity(y_score_oneshot, y_true)

####################################################################
# Loading all the weights

W = []
W1 = model.state_dict()['main.0.weight'].numpy()
B1 = model.state_dict()['main.0.bias'].numpy()
W2 = model.state_dict()['main.2.weight'].numpy()
B2 = model.state_dict()['main.2.bias'].numpy()
W3 = model.state_dict()['main.4.weight'].numpy()
B3 = model.state_dict()['main.4.bias'].numpy()
W.append(np.concatenate((W1,B1.reshape((-1, 1))), axis=1))
W.append(np.concatenate((W2,B2.reshape((-1, 1))), axis=1))
W.append(np.concatenate((W3,B3.reshape((-1, 1))), axis=1))

print(np.max(np.abs(W1)))
print(np.max(np.abs(W2)))
print(np.max(np.abs(W3)))

# Plotting the ditsibution of the trained model's parameters
fig = plt.figure(figsize=(5, 4), layout="constrained")
ax = fig.subplots(1, 1, sharex=True, sharey=True)
bins = np.arange(-2.005,2.005,0.01)
dist0 = ax.hist(W[0].reshape(-1, 1), bins=bins, density=True, cumulative=True, histtype="step", label='$W^{(0)}$')
dist1 = ax.hist(W[1].reshape(-1, 1), bins=bins, density=True, cumulative=True, histtype="step", label='$W^{(1)}$')
dist2 = ax.hist(W[2].reshape(-1, 1), bins=bins, density=True, cumulative=True, histtype="step", label='$W^{(2)}$')
plt.ylabel('Empirical PDF')
plt.legend()
plt.grid()
plt.xlim(-2,2)
plt.savefig(FIG_SAVE_PATH + "distribution_parameters.png" )


####################################################################
# Experiments on one-shot inference with 3-bit resolution pixels

H0 = np.concatenate((X_test_Q,np.ones((N_TEST, 1))), axis=1)
H1 = np.matmul(H0, np.transpose(W[0]))
H1 [H1 < 0] = 0
H1 = np.concatenate((H1,np.ones((N_TEST, 1))), axis=1)
H2 = np.matmul(H1, np.transpose(W[1]))
H2 [H2 < 0] = 0
H2 = np.concatenate((H2,np.ones((N_TEST, 1))), axis=1)
H3 = np.matmul(H2, np.transpose(W[2])).squeeze(-1)
y_score_oneshot_Q = sigmoid(H3)
[fpr_oneshot_Q, tpr_oneshot_Q, auc_oneshot_Q, acc_th05_oneshot_Q] = utlity(y_score_oneshot_Q, y_true)

####################################################################
# Layered-resolution inference

H_init = [np.zeros((N_TEST,N_FEATURES)), np.zeros((N_TEST,NUM_HIDDEN_NODES)), np.zeros((N_TEST,NUM_HIDDEN_NODES)), np.zeros((N_TEST, N_OUT))] 
A_init = [np.zeros_like(W[0]), np.zeros_like(W[1]), np.zeros_like(W[2])]

P_X = [0,-1,-2,-3,-4]
P_W0 = [-1,-2,-3,-4,-5]
P_W1 = [0,-1,-2,-3,-4]
P_W2 = [1,0,-1,-2,-3]
num_res = len(P_W0)-1

delta_X = partition(X_test.numpy(),P_X)
delta_W0 = partition(W[0],P_W0)
delta_W1 = partition(W[1],P_W1)
delta_W2 = partition(W[2],P_W2)

fpr_layered = []
tpr_layered = []
auc_layered = []
acc_th05_layered = []
norm_squared_errors = []
y_score_layered = []
sample1_activations_H1 = np.zeros((0,NUM_HIDDEN_NODES))
sample1_activations_H2 = np.zeros((0,NUM_HIDDEN_NODES))

for r in np.arange(num_res):
    delta_H0 = delta_X[r] * (2**P_X[r+1])
    delta_A = [delta_W0[r]*(2**P_W0[r+1]), delta_W1[r]*(2**P_W1[r+1]), delta_W2[r]*(2**P_W2[r+1])]
    y_score = evaluation_update(H_init, A_init, delta_H0, delta_A, np.ones(N_TEST,dtype=bool), -4)
    [fpr, tpr, auc, acc_th05] = utlity(y_score, y_true)
    y_score_layered.append(y_score)
    fpr_layered.append(fpr)
    tpr_layered.append(tpr)
    auc_layered.append(auc)
    acc_th05_layered.append(acc_th05)
    norm_squared_errors.append(np.square(y_score - y_score_oneshot))
    sample1_activations_H1 = np.vstack ((sample1_activations_H1, H_init[1][0,:]))
    sample1_activations_H2 = np.vstack ((sample1_activations_H2, H_init[2][0,:]))

####################################################################
# Adaptive resolution startegy

# Defining the gray zone
gray_area_max = 0.6
gray_area_min = 0.3

H_init = [np.zeros((N_TEST,N_FEATURES)), np.zeros((N_TEST,NUM_HIDDEN_NODES)), np.zeros((N_TEST,NUM_HIDDEN_NODES)), np.zeros((N_TEST, N_OUT))] 
A_init = [np.zeros_like(W[0]), np.zeros_like(W[1]), np.zeros_like(W[2])]

P_X = [0,-1,-2,-3,-4]
P_W0 = [-1,-2,-3,-4,-5]
P_W1 = [0,-1,-2,-3,-4]
P_W2 = [1,0,-1,-2,-3]
num_res = len(P_W0)-1

delta_X = partition(X_test.numpy(),P_X)
delta_W0 = partition(W[0],P_W0)
delta_W1 = partition(W[1],P_W1)
delta_W2 = partition(W[2],P_W2)

ind = np.ones(N_TEST,dtype=bool)
upgrade_needed = [ind]
y_score = np.ones(N_TEST)*0.5

for r in np.arange(num_res):
    delta_H0 = delta_X[r] * (2**P_X[r+1])
    delta_A = [delta_W0[r]*(2**P_W0[r+1]), delta_W1[r]*(2**P_W1[r+1]), delta_W2[r]*(2**P_W2[r+1])]
    y_score[ind] = evaluation_update(H_init, A_init, delta_H0, delta_A, ind, -4)
    ind = ( y_score >= gray_area_min ) & (y_score <= gray_area_max )
    upgrade_needed.append(ind)

[fpr_adaptive, tpr_adaptive, auc_adaptive, acc_th05_adaptive] = utlity(y_score, y_true)    
print(upgrade_needed)

####################################################################
# Printing ROC AUC
for r in np.arange(num_res):
    print(auc_layered[r])
print(auc_adaptive)
print(auc_oneshot_Q)
print(auc_oneshot)

####################################################################
# Printing prediction accuracy
for r in np.arange(num_res):
    print(acc_th05_layered[r])
print(acc_th05_adaptive)
print(acc_th05_oneshot_Q)
print(acc_th05_oneshot)

####################################################################
# Plotting ROC curve
plt.figure()
for r in np.arange(num_res):
    plt.plot(fpr_layered[r], tpr_layered[r], label=f'Resolution {r+1}')
# plt.plot(fpr_oneshot_Q, tpr_oneshot_Q, label='3-bit resolution pixels, full resolution weights')
plt.plot(fpr_adaptive, tpr_adaptive, label='Adaptive resolution',linestyle='dashed',color='black')
plt.plot(fpr_oneshot, tpr_oneshot, label='One-shot resolution')
plt.xlabel('False Positive Rate',fontsize = 17)
plt.ylabel('True Positive Rate' ,fontsize = 17)
plt.legend(fontsize = 17)
plt.grid()
plt.savefig(FIG_SAVE_PATH+'ROC.png')

np.savetxt('draft0.txt', fpr_adaptive,delimiter='\n')
np.savetxt('draft1.txt', tpr_adaptive,delimiter='\n')




####################################################################
# Plotting cdf for normalized squared errors
fig = plt.figure(figsize=(5, 4), layout="constrained")
ax = fig.subplots(1, 1, sharex=True, sharey=True)
bins = 10**(np.arange(-12,0,0.01))
for r in np.arange(num_res):
    ax.hist(norm_squared_errors[r], bins=bins, density=True, histtype="step", cumulative=True, label=f'Resolution {r+1}')
plt.xscale('log')
plt.ylabel('Empirical CDF')
plt.xlabel('normalized square of output errors')
plt.legend()
plt.grid()
plt.xlim(0, 1)
plt.savefig(FIG_SAVE_PATH + "NMSE_log.png" )

####################################################################
# Printing Empirical CDF for model output at different resolution upgrades
fig = plt.figure(figsize=(5, 4), layout="constrained")
ax = fig.subplots(1, 1, sharex=True, sharey=True)
bins = np.arange(0,1.1,0.001)
for r in np.arange(num_res):
    ax.hist(y_score_layered[r], bins=bins, density=True, histtype="step", cumulative=True, label=f'Resolution {r+1}')
plt.ylabel('Empirical CDF')
plt.xlabel('Output')
plt.legend()
plt.grid()
plt.xlim(0,1)
plt.savefig(FIG_SAVE_PATH + "Dist_output.png" )


####################################################################
# Printing activation status for sample 0
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,12))
ax1.imshow(np.transpose(sample1_activations_H1))
ax2.imshow(np.transpose(sample1_activations_H2))
ax1.set_title('Layer 0',fontsize = 40)
ax2.set_title('Layer 1',fontsize = 40)
ax1.axis('off')
ax2.axis('off')
plt.savefig(FIG_SAVE_PATH + "image 0, actiovations.png" )

####################################################################
# Printing number of samples that are in gray zone after each resolution upgrade
ax = plt.figure()
bins = [0,1,2,3,4]
for r in bins:
    upgrade_needed[r] = np.sum(upgrade_needed[r]==True)/N_TEST
plt.bar(bins,upgrade_needed)
for r in bins:
    plt.text(bins[r], upgrade_needed[r], upgrade_needed[r], ha = 'center')
plt.xlabel('Resolution Layer')
plt.ylabel('Ratio of samples in gray zone' )
plt.savefig(FIG_SAVE_PATH+'hist_adaptive.png')
