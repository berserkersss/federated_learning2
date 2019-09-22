#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        img_idx = self.idxs[item]
        return image, label, img_idx


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        img_train_list = []
        all_x_list = []

        for batch_idx, (images, labels, img_idxs) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)  # predicted label
            loss = self.loss_func(log_probs, labels)
            loss.retain_grad()  # compute the gradient of batch loss
            log_probs.retain_grad()  # compute the gradienr of loss
            loss.backward()
            optimizer.step()

        return net.state_dict(), loss.item(), net.layer_input.weight.grad


class CLUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cl_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=int(len(DatasetSplit(dataset, idxs))), shuffle=True)

    def cltrain(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss_g = []

        for batch_idx, (images, labels, img_idxs) in enumerate(self.cl_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            # print(images.shape)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
        
        return net.state_dict(), loss.item(), net.layer_input.weight.grad


class LocalCLUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        for batch_idx, (images, labels, img_idxs) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)  # predicted label
            loss = self.loss_func(log_probs, labels)
            loss.retain_grad()  # compute the gradient of batch loss
            log_probs.retain_grad()  # compute the gradienr of loss
            loss.backward()
            optimizer.step()

        return net.layer_input.weight.grad