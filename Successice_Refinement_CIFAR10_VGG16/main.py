#######################################################################################################
# Midel Inference with Successive Refinement
# Author(s): Homa Esfahanizadeh
# @ MIT and Nokia Bell Labs
# Last Update: 07/26/2024
# This version is related to the application of the solution on VGG16+CIFAR10
#######################################################################################################

import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as LS
import torchvision.transforms as transforms

from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from jsonargparse import ArgumentParser

from models import vgg16
from data.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt

#####################################################################
# Action item:
# improve imeplementation for repository

#####################################################################
# # Check if the maximum values are close to each other
def are_max_values_far(tensor, threshold = 0.1):
    sorted_values, _ = torch.sort(tensor) # Sort tensor in ascending order
    max1 = sorted_values[:,-1]
    max2 = sorted_values[:,-2]
    diff = max1 - max2
    return diff > threshold

#####################################################################
# Partitioning each element of an array into multiple smaller numbers
# Example: for num = -13.625 and p_val = [4,0,-3], 
#           num_parts = [-13.0, -5.0]
#           This is because -13.625 =  (-13*(2**0) - 5*(2**-3))
def partition(num, p_val):
    num_sign = torch.sign(num)
    num_abs = torch.absolute(num)
    num_abs = torch.fmod (num_abs / (2**p_val[0]),1)
    num_parts = []
    
    for i in np.arange(len(p_val)-1):
        num_shifted = num_abs * (2**(p_val[i]-p_val[i+1]))
        num_parts.append(torch.floor(num_shifted)*num_sign) 
        num_abs = torch.fmod (num_shifted,1)
    return num_parts

#####################################################################
# ketnel size 2 and stride size 2 -- for now -- later needs to be generalized
def MaxPool2D_oneshot(X):
    Y = torch.zeros((X.size(0),X.size(1),X.size(2)//2,X.size(3)//2)).to(X.device)
    kernel = 2 # assiming kernel = stride
    for i in np.arange(X.size(2)//kernel):
        for j in np.arange(X.size(3)//kernel):
            window = X[:,:,i*kernel:(i+1)*kernel,j*kernel:(j+1)*kernel]
            window = window.max(dim=2)[0]
            window = window.max(dim=2)[0]
            Y[:,:,i,j] = window
    return Y

#####################################################################
def relu_oneshot(X):
    output = torch.zeros_like(X).to(X.device)
    output[ X > 0 ] = X[ X > 0 ]
    return output

def relu_layered(X,Y):
    output = torch.zeros_like(X).to(X.device)
    output[ (X < 0) & ((X+Y) < 0) ] = 0
    output[ (X < 0) & ((X+Y) >= 0) ] = X[(X<0)&((X+Y)>=0)]+Y[(X < 0)&((X+Y) >= 0)]
    output[ (X >= 0) & ((X+Y) < 0) ] = X[(X>=0)&((X+Y)<0)]*-1
    output[ (X >= 0) & ((X+Y) >= 0) ] = Y[(X>=0)&((X+Y)>=0)]
    return output

#####################################################################
def plot_model_distribution (dataset, model):

    model_params = []
    for param_tensor_name in model.state_dict():
        param = model.state_dict()[param_tensor_name]
        model_params.append(np.array(param.cpu().flatten()))
    
    for ind in np.arange(16):

        plt.figure()
        plt.rcParams.update({'font.size': 18})
        plt.xlabel('Values')
        plt.ylabel('Frequency')
            
        plt.hist(model_params[ind*2], weights= np.ones_like(model_params[ind*2])/np.ones_like(model_params[ind*2]).size, bins=30, color='skyblue', alpha= 0.6, label='weight')
        plt.hist(model_params[ind*2+1], weights= np.ones_like(model_params[ind*2+1])/np.ones_like(model_params[ind*2+1]).size, bins=30, color='red', alpha= 0.4, label='bias')
        
        if (ind < 13):
            plt.title(f'Cnvolutional layer {ind + 1}')
        else:
            plt.title(f'Dense layer {ind - 13 + 1}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Figures/CIFAR10_VGG/Weight_DIST/{ind}.png')
    plt.close('all')

    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.hist(dataset, weights= np.ones_like(dataset)/np.ones_like(dataset).size, bins=30, color='green', alpha= 0.6)
    plt.title('input pixels')
    plt.tight_layout()
    plt.savefig(f'Figures/CIFAR10_VGG/Input_DIST.png')

#####################################################################
def one_shot_inference_default(dataset, model, device):
    
    criterion = nn.CrossEntropyLoss()

    torch.set_grad_enabled(False)
    model.eval()
    metric_hist = {
        'loss': [],
        'accuracy': [],
    }

    with tqdm(dataset, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as dataloader:
        for _, (imgs, labels) in enumerate(dataloader):
            pbar_desc = f'evaluation'
            dataloader.set_description(pbar_desc)
            epoch_postfix = OrderedDict()
            
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            _, predictions = torch.max(outputs.data, 1)
            correct_predictions = (predictions == labels).sum() / predictions.size(0)

            metric_hist['loss'].append(loss.item())
            epoch_postfix['loss'] = loss.item()
            metric_hist['accuracy'].append(correct_predictions.item())
            
            dataloader.set_postfix(**epoch_postfix)
            
    metric_hist = {k: np.nanmean(v) for k, v in metric_hist.items()}

    stats_str = 'Summary ({}): epoch {} | '.format(job_name, cp['epoch'])
    for k, v in metric_hist.items():
        stats_str += '{}: {:.5f} | '.format(k, v)
    print(stats_str)

#####################################################################
def one_shot_inference(dataset, model, device):

    criterion = nn.CrossEntropyLoss()
    metric_hist = {
        'loss': [],
        'accuracy': [],
    }

    model_params = []
    for param_tensor_name in model.state_dict():
        param = model.state_dict()[param_tensor_name]
        param = param.cpu()
        param = torch.tensor(param.clone().detach()).to(device)
        model_params.append(param)

    with tqdm(dataset, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as dataloader:
        for _, (imgs, labels) in enumerate(dataloader):
            pbar_desc = f'evaluation'
            dataloader.set_description(pbar_desc)
            epoch_postfix = OrderedDict()
            
            imgs, labels = imgs.to(device), labels.to(device)
            conv2d = nn.Conv2d(64, 3, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[0])
            conv2d.bias = nn.Parameter(model_params[1])
            output = conv2d(imgs)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[2])
            conv2d.bias = nn.Parameter(model_params[3])
            output = conv2d(output)
            output = relu_oneshot(output)
            output = MaxPool2D_oneshot(output)
            conv2d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[4])
            conv2d.bias = nn.Parameter(model_params[5])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[6])
            conv2d.bias = nn.Parameter(model_params[7])
            output = conv2d(output)
            output = relu_oneshot(output)
            output = MaxPool2D_oneshot(output)
            conv2d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[8])
            conv2d.bias = nn.Parameter(model_params[9])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[10])
            conv2d.bias = nn.Parameter(model_params[11])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[12])
            conv2d.bias = nn.Parameter(model_params[13])
            output = conv2d(output)
            output = relu_oneshot(output)
            output = MaxPool2D_oneshot(output)
            conv2d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[14])
            conv2d.bias = nn.Parameter(model_params[15])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[16])
            conv2d.bias = nn.Parameter(model_params[17])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[18])
            conv2d.bias = nn.Parameter(model_params[19])
            output = conv2d(output)
            output = relu_oneshot(output)
            output = MaxPool2D_oneshot(output)
            conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[20])
            conv2d.bias = nn.Parameter(model_params[21])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[22])
            conv2d.bias = nn.Parameter(model_params[23])
            output = conv2d(output)
            output = relu_oneshot(output)
            conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(model_params[24])
            conv2d.bias = nn.Parameter(model_params[25])
            output = conv2d(output)
            output = relu_oneshot(output)
            output = MaxPool2D_oneshot(output)
            output = output.view(output.size(0), -1)
            linear_layer = nn.Linear(512, 512)
            linear_layer.weight = nn.Parameter(model_params[26])
            linear_layer.bias = nn.Parameter(model_params[27])
            output = linear_layer(output)
            output = relu_oneshot(output)
            linear_layer = nn.Linear(512, 512)
            linear_layer.weight = nn.Parameter(model_params[28])
            linear_layer.bias = nn.Parameter(model_params[29])
            output = linear_layer(output)
            output = relu_oneshot(output)
            linear_layer = nn.Linear(512, 10)
            linear_layer.weight = nn.Parameter(model_params[30])
            linear_layer.bias = nn.Parameter(model_params[31])
            output = linear_layer(output)
            
            loss = criterion(output, labels)            
            _, predictions = torch.max(output.data, 1)
            correct_predictions = (predictions == labels).sum() / predictions.size(0)

            metric_hist['loss'].append(loss.item())
            epoch_postfix['loss'] = loss.item()
            metric_hist['accuracy'].append(correct_predictions.item())
            
            dataloader.set_postfix(**epoch_postfix)
            
    metric_hist = {k: np.nanmean(v) for k, v in metric_hist.items()}

    stats_str = 'Summary ({}): epoch {} | '.format(job_name, cp['epoch'])
    for k, v in metric_hist.items():
        stats_str += '{}: {:.5f} | '.format(k, v)
    print(stats_str)

#####################################################################
# Updated evaluation of a neural network, given the weight and samle changes
def evaluation_update(H, A, delta_H0, delta_A, H_limit=-6):

    hidden_layer = H[0]
    delta_hidden_layer = torch.round(delta_H0 / (2**H_limit)) * (2**H_limit)
    H[0] = H[0] + delta_H0

    dims = [(3,64),(64,64),(64,128),(128,128),(128,256),(256,256),(256,256),(256,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512),(10,512)]
    
    for l in np.arange(1,17,1):
        if l in [1,3,5,6,8,9,11,12]: # concolutional layer
            conv2d = nn.Conv2d(dims[l-1][1], dims[l-1][0], kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(A[2*(l-1)])
            conv2d.bias = nn.Parameter(torch.zeros_like(A[2*(l-1)+1]))
            delta = conv2d(delta_hidden_layer)
            conv2d.weight = nn.Parameter(delta_A[2*(l-1)])
            conv2d.bias = nn.Parameter(delta_A[2*(l-1)+1])
            delta = delta + conv2d(hidden_layer+delta_hidden_layer)
            delta = torch.round(delta / (2**H_limit)) * (2**H_limit)
            hidden_layer = relu_oneshot(H[l])
            delta_hidden_layer = relu_layered(H[l],delta)
            A[2*(l-1)] = A[2*(l-1)] + delta_A[2*(l-1)]
            A[2*(l-1)+1] = A[2*(l-1)+1] + delta_A[2*(l-1)+1]
            H[l] = H[l] + delta
        elif  l in [2,4,7,10,13]: # convolutional layer with maxpool
            conv2d = nn.Conv2d(dims[l-1][1], dims[l-1][0], kernel_size=3, padding=1)
            conv2d.weight = nn.Parameter(A[2*(l-1)])
            conv2d.bias = nn.Parameter(torch.zeros_like(A[2*(l-1)+1]))
            delta = conv2d(delta_hidden_layer)
            conv2d.weight = nn.Parameter(delta_A[2*(l-1)])
            conv2d.bias = nn.Parameter(delta_A[2*(l-1)+1])
            delta = delta + conv2d(hidden_layer+delta_hidden_layer)
            delta = torch.round(delta / (2**H_limit)) * (2**H_limit)
            hidden_layer = relu_oneshot(H[l])
            delta_hidden_layer = relu_layered(H[l],delta)
            delta_hidden_layer = MaxPool2D_oneshot ( hidden_layer + delta_hidden_layer ) - MaxPool2D_oneshot ( hidden_layer )
            hidden_layer = MaxPool2D_oneshot( hidden_layer)
            A[2*(l-1)] = A[2*(l-1)] + delta_A[2*(l-1)]
            A[2*(l-1)+1] = A[2*(l-1)+1] + delta_A[2*(l-1)+1]
            H[l] = H[l] + delta
        else: # dense layer
            if l == 14:
                hidden_layer = hidden_layer.view(hidden_layer.size(0),-1)
                delta_hidden_layer = delta_hidden_layer.view(delta_hidden_layer.size(0), -1)
            linear_layer = nn.Linear(dims[l-1][0], dims[l-1][1])
            linear_layer.weight = nn.Parameter(A[2*(l-1)])
            linear_layer.bias = nn.Parameter(torch.zeros_like(A[2*(l-1)+1]))
            delta = linear_layer(delta_hidden_layer)
            linear_layer.weight = nn.Parameter(delta_A[2*(l-1)])
            linear_layer.bias = nn.Parameter(delta_A[2*(l-1)+1])
            delta = delta + linear_layer(hidden_layer+delta_hidden_layer)
            delta = torch.round(delta / (2**H_limit)) * (2**H_limit)
            if ( l < 16):
                hidden_layer = relu_oneshot(H[l])
                delta_hidden_layer = relu_layered(H[l],delta)
            A[2*(l-1)] = A[2*(l-1)] + delta_A[2*(l-1)]
            A[2*(l-1)+1] = A[2*(l-1)+1] + delta_A[2*(l-1)+1]
            H[l] = H[l] + delta
    return H


#####################################################################
device = 'cuda:7'
job_name = '2024-06-17 16:39:47' # name of the trained model
save_path = 'checkpoints' # location of the trained model

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Inference for Cifar10 + VGG 16
# Create CIFAR-10 dataset and dataloader
batch_size = 512
dataset_path ='data/cifar-10-batches-py'
test_dataset = CIFAR10(dataset_path, 'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the VGG16 trained model
model = vgg16()
model.to(device)
save_path = save_path + '/' + job_name
cp = torch.load('{}/checkpoint_latest.pth'.format(save_path), map_location='cpu')
model.load_state_dict(cp['model'])
print('Loaded weights from epoch {}'.format(cp['epoch']))
# print(model.state_dict())

# one_shot_inference_default(test_loader, model, device)
# plot_model_distribution (model)
# ind = 1
# for param_tensor_name in model.state_dict():
#     param = model.state_dict()[param_tensor_name]
#     print(ind, math.log2(max(param.max(),-param.min())))
#     ind = ind + 1

# one_shot_inference(test_loader, model, device)

####################################################################
# Layered-resolution inference

imgs, labels = next(iter(test_loader)) # considering the first batch for the evaluation with successive refinement
imgs = imgs.to(device)
labels = labels.to(device)

criterion = nn.CrossEntropyLoss()

H_init = [torch.zeros(batch_size,3,32,32).to(device),
          torch.zeros(batch_size,64,32,32).to(device),
          torch.zeros(batch_size,64,32,32).to(device), 
          torch.zeros(batch_size,128,16,16).to(device),
          torch.zeros(batch_size,128,16,16).to(device),
          torch.zeros(batch_size,256,8,8).to(device),
          torch.zeros(batch_size,256,8,8).to(device), 
          torch.zeros(batch_size,256,8,8).to(device),
          torch.zeros(batch_size,512,4,4).to(device),
          torch.zeros(batch_size,512,4,4).to(device),
          torch.zeros(batch_size,512,4,4).to(device),
          torch.zeros(batch_size,512,2,2).to(device),
          torch.zeros(batch_size,512,2,2).to(device),
          torch.zeros(batch_size,512,2,2).to(device), 
          torch.zeros(batch_size,512).to(device),
          torch.zeros(batch_size,512).to(device),
          torch.zeros(batch_size,10).to(device)]

A_init = []
for param_tensor_name in model.state_dict():
    param = model.state_dict()[param_tensor_name]
    A_init.append(torch.zeros_like(param))

# P_X = [2,-2,-3,-4,-10] # The maximum value absolute value for pixels is 2.6400
# P_W = []
# P_W.append ([-1,-5,-6,-7,-13])
# P_W.append ([-1,-5,-6,-7,-13])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-1,-5,-6,-7,-13])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-1,-5,-6,-7,-13])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-4,-8,-9,-10,-16])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-5,-9,-10,-11,-17])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-4,-8,-9,-10,-16])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-5,-9,-10,-11,-17])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-5,-9,-10,-11,-17])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-3,-7,-8,-9,-15])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([-2,-6,-7,-8,-14])
# P_W.append ([0,-4,-8,-9,-15])


weight_max_order = [-1,-1,-2,-1,-2,-1,-2,-2,-2,-2,-3,-3,-2,-3,-3,-3,-3,-4,-3,-5,-3,-4,-3,-5,-3,-5,-3,-3,-3,-2,-2,0]
num_res = 7
P_X = [2]
for r in np.arange(num_res+1):
    P_X.append(P_X[-1]-1)
P_W = []
for l in np.arange(32):
    P_W_l = [weight_max_order[l]]
    for r in np.arange(num_res+1):
        P_W_l.append(P_W_l[-1]-1)
    P_W.append(P_W_l)

# P_X = [2,2-cut_off_1,2-cut_off_2,2-cut_off_3,2-cut_off_4]  # The maximum value absolute value for pixels is 2.6400
# P_W = []
# for l in np.arange(32):
#     P_W.append ( [weight_max_order[l],weight_max_order[l]-cut_off_1,weight_max_order[l]-cut_off_2,weight_max_order[l]-cut_off_3,weight_max_order[l]-cut_off_4] )

# num_res = len(P_X)-1
# num_res = 1
# print(num_res)

delta_X = partition(imgs.cpu(),P_X)
delta_W = []
for res in np.arange(num_res):
    delta_W_res = []
    l = 0
    for param_tensor_name in model.state_dict():
        param = model.state_dict()[param_tensor_name]
        param = param.cpu()
        param = partition(param, P_W[l])
        delta_W_res.append(param[res])
        l = l + 1
    delta_W.append(delta_W_res)

metric_hist = {
    'loss': [],
    'accuracy': [],
}

for res in np.arange(num_res):
    delta_H0 = delta_X[res] * (2**P_X[res+1])
    delta_H0 = delta_H0.to(device)
    
    delta_A = []
    for l in np.arange(32):
        param = delta_W[res][l] * (2**P_W[l][res+1])
        param = param.to(device)
        delta_A.append ( param )
        l = l + 1

    H_init = evaluation_update(H_init, A_init, delta_H0, delta_A)

    output = H_init[16]
    loss = criterion(output, labels)            
    _, predictions = torch.max(output.data, 1)
    correct_predictions = (predictions == labels).sum() / predictions.size(0)
    metric_hist['loss'].append(loss.item())
    metric_hist['accuracy'].append(correct_predictions.item())
    # print(H_init[16])
    print(res, correct_predictions.item(), loss.item())

# plt.figure()
# plt.plot(np.arange(1,num_res+1,1), metric_hist['accuracy'],'-o', color = 'blue', label='layering strategy')
# plt.plot(np.arange(1,num_res+1,1), np.ones((num_res)) * 0.895,'--', color = 'blue', label='one-shot strategy')
# plt.xlabel('Resolution upgrade',fontsize = 17)
# plt.ylabel('Accuracy' ,fontsize = 17)
# plt.legend(fontsize = 17)
# plt.grid()
# plt.savefig('Figures/accuracy_multi_res.png')

# # Define data
# fig, ax1 = plt.subplots()
# ax1.set_xlabel('Resolution upgrade',fontsize = 17)
# ax1.set_ylabel('Generalization accuracy' ,fontsize = 17, color='blue')
# ax1.plot(np.arange(1,num_res+1,1), metric_hist['accuracy'],'-o', color = 'blue', label='layering strategy')
# ax1.plot(np.arange(1,num_res+1,1), np.ones((num_res)) * 0.895,'--', color = 'blue', label='one-shot strategy')
# # Adding twin axes
# ax2 = ax1.twinx()
# ax2.set_ylabel('Cross entropy loss' ,fontsize = 17, color='red')
# plt.plot(np.arange(1,num_res+1,1), metric_hist['loss'],'-o', color = 'red')
# plt.plot(np.arange(1,num_res+1,1), np.ones((num_res)) * 0.895,'--', color = 'red')

# # Show the plot
# plt.savefig('Figures/test.png')

####################################################################
# Adaptive-resolution inference

imgs, labels = next(iter(test_loader)) # considering the first batch for the evaluation with successive refinement
imgs = imgs.to(device)
labels = labels.to(device)

criterion = nn.CrossEntropyLoss()

torch.set_grad_enabled(False)

H_init = [torch.zeros(batch_size,3,32,32).to(device),
          torch.zeros(batch_size,64,32,32).to(device),
          torch.zeros(batch_size,64,32,32).to(device), 
          torch.zeros(batch_size,128,16,16).to(device),
          torch.zeros(batch_size,128,16,16).to(device),
          torch.zeros(batch_size,256,8,8).to(device),
          torch.zeros(batch_size,256,8,8).to(device), 
          torch.zeros(batch_size,256,8,8).to(device),
          torch.zeros(batch_size,512,4,4).to(device),
          torch.zeros(batch_size,512,4,4).to(device),
          torch.zeros(batch_size,512,4,4).to(device),
          torch.zeros(batch_size,512,2,2).to(device),
          torch.zeros(batch_size,512,2,2).to(device),
          torch.zeros(batch_size,512,2,2).to(device), 
          torch.zeros(batch_size,512).to(device),
          torch.zeros(batch_size,512).to(device),
          torch.zeros(batch_size,10).to(device)]

A_init = []
for param_tensor_name in model.state_dict():
    param = model.state_dict()[param_tensor_name]
    A_init.append(torch.zeros_like(param))

weight_max_order = [-1,-1,-2,-1,-2,-1,-2,-2,-2,-2,-3,-3,-2,-3,-3,-3,-3,-4,-3,-5,-3,-4,-3,-5,-3,-5,-3,-3,-3,-2,-2,0]
num_res = 7
P_X = [2]
for r in np.arange(num_res+1):
    P_X.append(P_X[-1]-1)
P_W = []
for l in np.arange(32):
    P_W_l = [weight_max_order[l]]
    for r in np.arange(num_res+1):
        P_W_l.append(P_W_l[-1]-1)
    P_W.append(P_W_l)

delta_X = partition(imgs.cpu(),P_X)
delta_W = []
for res in np.arange(num_res):
    delta_W_res = []
    l = 0
    for param_tensor_name in model.state_dict():
        param = model.state_dict()[param_tensor_name]
        param = param.cpu()
        param = partition(param, P_W[l])
        delta_W_res.append(param[res])
        l = l + 1
    delta_W.append(delta_W_res)

num_samples_grey_zone = []
output = torch.zeros_like(H_init[16])

for res in np.arange(num_res):
    delta_H0 = delta_X[res] * (2**P_X[res+1])
    delta_H0 = delta_H0.to(device)
    
    delta_A = []
    for l in np.arange(32):
        param = delta_W[res][l] * (2**P_W[l][res+1])
        param = param.to(device)
        delta_A.append ( param )
        l = l + 1

    H_init = evaluation_update(H_init, A_init, delta_H0, delta_A)
    resolved_samples = are_max_values_far(H_init[16], threshold = 0.05).cpu()
    output[resolved_samples,:] = H_init[16][resolved_samples,:]
    num_samples_grey_zone.append(batch_size-sum(resolved_samples))

loss = criterion(output, labels)            
_, predictions = torch.max(output.data, 1)
correct_predictions = (predictions == labels).sum() / predictions.size(0)
metric_hist_loss = loss.item()
metric_hist_acc = correct_predictions.item()
print('******************************************')
print(res, correct_predictions.item(), loss.item())

plt.figure()
plt.plot(np.arange(1,num_res+1,1), metric_hist['accuracy'],'-o', color = 'blue', label='layering strategy')
plt.plot(np.arange(1,num_res+1,1), np.ones((num_res)) * metric_hist_acc,'--', color = 'green', label='adaptive strategy')
plt.plot(np.arange(1,num_res+1,1), np.ones((num_res)) * 0.894,'--', color = 'black', label='one-shot strategy')
plt.xlabel('Resolution upgrade',fontsize = 17)
plt.ylabel('Accuracy' ,fontsize = 17)
plt.legend(fontsize = 17)
plt.grid()
plt.savefig('Figures/accuracy_multi_res.png')

# Printing number of samples that are in gray zone after each resolution upgrade
ax = plt.figure()
bins = np.arange(1,num_res+1,1)
num_samples_grey_zone = [round(p.item() * 100 / batch_size)/100 for p in num_samples_grey_zone]
num_samples_grey_zone[0] = 1.0
# num_samples_grey_zone[0] = batch_size
plt.bar(bins,num_samples_grey_zone)
for r in bins:
    plt.text(bins[r-1], num_samples_grey_zone[r-1], num_samples_grey_zone[r-1], ha = 'center',fontsize = 13)
plt.xlabel('Resolution upgrade',fontsize = 17)
plt.ylabel('Ratio of samples in gray zone',fontsize = 17)
plt.savefig('Figures/hist_adaptive_cifar.png')

# plot_model_distribution (imgs.cpu().flatten(), model)

