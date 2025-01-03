import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as LS
import torchvision.transforms as transforms

import os
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from jsonargparse import ArgumentParser

from models import vgg16
from data.datasets import CIFAR10
import numpy as np

def train_classifier(job_name, Config):

    # Create CIFAR-10 train dataset and dataloader
    train_dataset = CIFAR10(Config.dataset.path, 'train')
    val_dataset = CIFAR10(Config.dataset.path, 'test')
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True)

    # Initialize the VGG16 model
    model = vgg16()
    model.to(Config.device)
    criterion = nn.CrossEntropyLoss()

    # Define loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer_scheduler = LS.MultiplicativeLR(optimizer, lr_lambda=lambda x: Config.scheduler.lr_schedule_factor)

    start_epoch = 0
    if Config.resume_job:
        save_path = Config.save.path + '/' + Config.resume_job
        cp = torch.load('{}/checkpoint_latest.pth'.format(save_path), map_location='cpu')
        model.load_state_dict(cp['model'])
        print('Loaded weights from epoch {}'.format(cp['epoch']))
        start_epoch = cp['epoch'] + 1
    
    best_accuracy = 0
    
    for epoch in range(start_epoch, Config.num_epochs):
        # training step

        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        lr = Config.learning_rate * (0.5 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        torch.set_grad_enabled(True)
        model.train()
        optimizer.zero_grad()
        metric_hist = {
        'loss': [],
        'accuracy': [],
        }
        correct_predictions = 0
        with tqdm(train_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as dataloader:
            for _, (imgs, labels) in enumerate(dataloader):
                pbar_desc = f'training, epoch {epoch},'
                dataloader.set_description(pbar_desc)
                epoch_postfix = OrderedDict()
                optimizer.zero_grad()
                
                imgs, labels = imgs.to(Config.device), labels.to(Config.device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predictions = torch.max(outputs.data, 1)
                correct_predictions = (predictions == labels).sum() / Config.batch_size

                metric_hist['loss'].append(loss.item())
                epoch_postfix['loss'] = loss.item()
                metric_hist['accuracy'].append(correct_predictions.item())
                
                dataloader.set_postfix(**epoch_postfix)
                
        metric_hist = {k: np.nanmean(v) for k, v in metric_hist.items()}

        stats_str = 'Summary ({}): epoch {} | '.format(job_name, epoch)
        for k, v in metric_hist.items():
            stats_str += '{}: {:.5f} | '.format(k, v)
        print(stats_str)

        # validation step
        torch.set_grad_enabled(False)
        model.eval()
        metric_hist = {
        'loss': [],
        'accuracy': [],
        }
        with tqdm(val_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as dataloader:
            for _, (imgs, labels) in enumerate(dataloader):
                pbar_desc = f'validation,'
                dataloader.set_description(pbar_desc)
                epoch_postfix = OrderedDict()
                optimizer.zero_grad()
                
                imgs, labels = imgs.to(Config.device), labels.to(Config.device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                _, predictions = torch.max(outputs.data, 1)
                correct_predictions = (predictions == labels).sum() / predictions.size(0)

                metric_hist['loss'].append(loss.item())
                epoch_postfix['loss'] = loss.item()
                metric_hist['accuracy'].append(correct_predictions.item())
                
                dataloader.set_postfix(**epoch_postfix)
                
        metric_hist = {k: np.nanmean(v) for k, v in metric_hist.items()}
        stats_str = 'Summary ({}): epoch {} | '.format(job_name, epoch)
        for k, v in metric_hist.items():
            stats_str += '{}: {:.5f} | '.format(k, v)
        print(stats_str)

        save_path = Config.save.path + '/' + job_name

        if (metric_hist['accuracy'] > best_accuracy ):
            best_accuracy = metric_hist['accuracy']
            print('Saving best weights')
            if not os.path.exists(save_path):
                print('Creating model directory: {}'.format(save_path))
                os.makedirs(save_path)
            torch.save({'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}, 
                       f'{save_path}/checkpoint_latest.pth')
        else:
            cp = torch.load('{}/checkpoint_latest.pth'.format(save_path), map_location='cpu')
            model.load_state_dict(cp['model'])
            print('loading weights from epoch {}'.format(cp['epoch']))

def test_classifier(job_name, Config):

    # Create CIFAR-10 dataset and dataloader
    test_dataset = CIFAR10(Config.dataset.path, 'test')
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True)

    # Initialize the VGG16 model
    model = vgg16()
    model.to(Config.device)
    criterion = nn.CrossEntropyLoss()

    save_path = Config.save.path + '/' + job_name
    cp = torch.load('{}/checkpoint_latest.pth'.format(save_path), map_location='cpu')
    model.load_state_dict(cp['model'])
    print('Loaded weights from epoch {}'.format(cp['epoch']))
    
    torch.set_grad_enabled(False)
    model.eval()
    metric_hist = {
    'loss': [],
    'accuracy': [],
    }
    with tqdm(test_loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as dataloader:
        for _, (imgs, labels) in enumerate(dataloader):
            pbar_desc = f'evaluation'
            dataloader.set_description(pbar_desc)
            epoch_postfix = OrderedDict()
            
            imgs, labels = imgs.to(Config.device), labels.to(Config.device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            _, predictions = torch.max(outputs.data, 1)
            correct_predictions = (predictions == labels).sum() / Config.batch_size

            metric_hist['loss'].append(loss.item())
            epoch_postfix['loss'] = loss.item()
            metric_hist['accuracy'].append(correct_predictions.item())
            
            dataloader.set_postfix(**epoch_postfix)
            
    metric_hist = {k: np.nanmean(v) for k, v in metric_hist.items()}

    stats_str = 'Summary ({}): epoch {} | '.format(job_name, cp['epoch'])
    for k, v in metric_hist.items():
        stats_str += '{}: {:.5f} | '.format(k, v)
    print(stats_str)

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, help='device to run job')
    parser.add_argument('--randomseed', type=int, help='random seed', default=1)
    parser.add_argument('--resume_job', type=str, help='resume training of job (encoder)')
    parser.add_argument('--eval_job', type=str, help='evaluate a trained encoder')
    parser.add_argument('--save.path', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--comments', type=str, help='some job description')

    parser.add_argument('--dataset.dataset', type=str, help='dataset: dataset to use')
    parser.add_argument('--dataset.path', type=str, default='data/cifar-10-batches-py', help='dataset: path to dataset')

    parser.add_argument('--batch_size', type=int, help='training batch size', default=16)
    parser.add_argument('--num_classes', type=int, help='number of classes', default=10)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--num_epochs', type=int, help='number of training epochs', default=200)
    
    parser.add_argument('--scheduler.lr_schedule_factor', type=float, help='scheduler: lr decrease factor', default=1.,)
    parser.add_argument('--scheduler.patience', type=int, help='scheduler: number of epochs to wait for improvement', default=1)

    Config = parser.parse_args()
    return Config


if __name__ == "__main__":
    Config = arg_parse()
    print(Config)

    seed = Config.randomseed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if(Config.eval_job):
        job_name = Config.eval_job
        test_classifier(job_name, Config)
    elif(Config.resume_job):
        job_name = Config.resume_job
        train_classifier(job_name, Config)
    else:
        job_name = str(datetime.now()).split('.')[0]
        train_classifier(job_name, Config)