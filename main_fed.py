#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 一层神经网络全连接，交叉熵训练模型

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import CLUpdate
from models.Update import LocalCLUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # sample users
        #if args.iid:
        dict_users_iid_temp = mnist_iid(dataset_train, args.num_users)
        #else:
        dict_users = mnist_noniid(dataset_train, args.num_users)

        #dict_users_iid_temp = dict_users
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    #print('img_size=',img_size)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob_cl = copy.deepcopy(net_glob)
    net_glob_fl = copy.deepcopy(net_glob)
    net_local_fl = [copy.deepcopy(net_glob) for i in range(args.num_users)]
    net_glob_fl.train()
    net_glob_cl.train()

    # copy weights
    w_glob_fl = net_glob_fl.state_dict()
    w_glob_cl = net_glob_cl.state_dict()


    # training
    eta = 0.01
    Nepoch = 5 # num of epoch 
    loss_train_fl, loss_train_cl = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    para_g = []
    loss_grad = []
    delta_batch_loss_list = []
    beta_list = []
    count_list = np.zeros(256).tolist()
    line1_iter_list = []
    line2_iter_list = []
    wgfed_list = []
    wgcl_list = []

    w_locals, loss_locals = [], []
    w0_locals,loss0_locals =[], []
    weight_div_list = []
    para_cl = []
    para_fl = []
    beta_locals, mu_locals, sigma_locals = [],[],[]
    x_stat_loacals, pxm_locals =[],[]
    data_locals = [[] for i in range(args.epochs)]
    w_fl_iter,w_cl_iter = [], []
    deltaloss_fl_iter, deltaloss_cl_iter = [], []
    beta_max_his, mu_max_his, sigma_max_his = [], [], []
    acc_train_cl_his, acc_train_fl_his = [], []

    net_glob_cl.eval()
    acc_train_cl, loss_train_clxx = test_img(net_glob_cl, dataset_train, args)
    acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
    acc_train_cl_his.append(acc_test_cl)
    acc_train_fl_his.append(acc_test_cl)
    print("Training accuracy: {:.2f}".format(acc_train_cl))
    print("Testing accuracy: {:.2f}".format(acc_test_cl))

    dict_users_iid = []
    for iter in range(args.num_users):
        dict_users_iid.extend(dict_users_iid_temp[iter])

    beta = -float("inf")
    lamb = float("inf")
    Ld = []


    for iter in range(args.epochs):  # num of iterations
        w_locals = []
        for iter_local in range(args.local_ep):
            # CL setting
            glob_cl = CLUpdate(args=args, dataset=dataset_train, idxs=dict_users_iid)
            w_cl, loss_cl, delta_loss_cl= glob_cl.cltrain(net=copy.deepcopy(net_glob_cl).to(args.device))
            net_glob_cl.load_state_dict(w_cl)  # update the CL w

            # FL setting
            # M clients local update
            m = args.num_users  # num of selected users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select randomly m clients
            Ld_temp = []
            for idx in idxs_users:
                glob_fl = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
                w_fl, loss, delta_loss_fl= glob_fl.train(net=copy.deepcopy(net_local_fl[idx]).to(args.device))
                if iter_local == args.local_ep - 1:
                    w_locals.append(copy.deepcopy(w_fl))  # collect local model
                net_local_fl[idx].load_state_dict(w_fl)  # update the FL w

                loss_locals.append(loss)  # collect local loss fucntion

                # Compute beta and lambda
                temp = torch.norm(delta_loss_cl - delta_loss_fl).item()/torch.norm(w_cl['layer_input.weight'] - w_fl['layer_input.weight']).item()
                if temp > beta:
                    beta = temp
                if temp < lamb:
                    lamb = temp

                loss_fl = sum(loss_locals) / len(loss_locals)

                # compute Ld
                local = LocalCLUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
                delta_loss = local.train(net=copy.deepcopy(net_glob_cl).to(args.device))
                Ld_temp.append(torch.norm(delta_loss_fl - delta_loss).item())

            Ld.append(np.mean(Ld_temp))

        # CL
        print('cl,iter = ', iter, 'loss=', loss_cl)
        # testing
        acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_cl))
        acc_train_cl_his.append(acc_test_cl.item())
        w_cl_iter.append(copy.deepcopy(net_glob_cl.state_dict()))

        # FedAvg
        w_glob_fl = FedAvg(w_locals)  # update the global model
        for idx in range(args.num_users):
            net_local_fl[idx].load_state_dict(w_glob_fl)
        net_glob_fl.load_state_dict(w_glob_fl)  # copy weight to net_glob
        w_fl_iter.append(copy.deepcopy(w_glob_fl))
        print('fl,iter = ', iter, 'loss=', loss_fl)

        # testing
        net_glob_fl.eval()
        acc_test_fl, loss_test_flxx = test_img(net_glob_fl, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_fl))
        acc_train_fl_his.append(acc_test_fl.item())

        print(beta)
        print(lamb)




    #weight divergence of simulation 
    for i in range(len(w_cl_iter)):
        para_cl = w_cl_iter[i]['layer_input.weight']
        para_fl = w_fl_iter[i]['layer_input.weight']
        line2 = torch.norm(para_cl-para_fl)
        line2_iter_list.append(line2.item())

    print('y_line2=',line2_iter_list) # simulation
    print('Ld=', Ld)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(line2_iter_list, c="red")
    plt.xlabel('Epochs')
    plt.ylabel('Difference')
    plt.savefig('Figure/different.png')


    colors = ["blue", "red"]
    labels = ["non-iid", "iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acc_train_fl_his, c=colors[0], label=labels[0])
    ax.plot(acc_train_cl_his, c=colors[1], label=labels[1])
    ax.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Figure/Accuracy_non_iid2_temp.png')

    line2_iter_list = pd.DataFrame(line2_iter_list)
    line2_iter_list.to_csv('csv/difference.csv')

    beta = pd.DataFrame(beta)
    beta.to_csv('csv/beta.csv')

    lamb = pd.DataFrame(lamb)
    lamb.to_csv('csv/lambda.csv')






