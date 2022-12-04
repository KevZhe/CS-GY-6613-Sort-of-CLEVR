"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
from model import CNN_MLP, CNN_RN_SOC, RN_state_desc


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['CNN_MLP', "CNN_RN_SOC", "RN_state_desc"], default='CNN_RN_SOC', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')
parser.add_argument('--dataset', type=str, default='pixels',
                    help='what dataset to train on. options: pixels, state_desc (default: pixels)')

#check for cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

summary_writer = SummaryWriter()

#select model based on arguments
if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
elif args.model=='RN_state_desc':
  model = RN_state_desc(args)
else:
  model = CNN_RN_SOC(args)



model_dirs = './model'
bs = args.batch_size

#create input tensor based on dataset to train on
if args.dataset == 'pixels':
    input_obj = torch.FloatTensor(bs, 3, 75, 75)
else:
    #state description tensor containing 6 objects and 7 features
    input_obj = torch.FloatTensor(bs, 6, 7)

input_qst = torch.FloatTensor(bs, 18)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_obj = input_obj.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_obj = Variable(input_obj)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    if args.dataset == "pixels":
        obj = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    else: #training on state description
        batch_data = data[0][bs*i:bs*(i+1)]
        #convert state description matrix into form usable by model
        new_data = []
        for matrix in batch_data:
            #for each state description matrix in batch
            new_matrix = []
            for row in matrix:
                #for each object in current matrix
                new_row = []
                for feature in row:
                    #for each feature in object, convert to value
                    if isinstance(feature, np.ndarray):
                        #append the x & y coordinates into our features list
                        new_row.extend(coordinate for coordinate in feature)
                    elif isinstance(feature, tuple):
                        #append the RGB values into our features list
                        new_row.extend(colorval/255 for colorval in feature)
                    elif isinstance(feature, str):
                        #append 1 if shape is rectangle, 0 if circle
                        if feature == 'rectangle': new_row.append(1)
                        else: new_row.append(0)
                    else:
                        #size feature
                        new_row.append(feature)
                new_matrix.append(new_row)
            new_data.append(new_matrix)
        obj = torch.from_numpy(np.asarray(new_data))
                    
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_obj.data.resize_(obj.size()).copy_(obj)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    obj = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (obj,qst,ans)

    
def train(epoch, ternary, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(ternary)
    random.shuffle(rel)
    random.shuffle(norel)

    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_ternary = []
    acc_rels = []
    acc_norels = []

    l_ternary = []
    l_binary = []
    l_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)

        accuracy_ternary, loss_ternary = model.train_(input_obj, input_qst, label)
        acc_ternary.append(accuracy_ternary.item())
        l_ternary.append(loss_ternary.item())

        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_obj, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_obj, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Ternary accuracy: {:.0f}% | Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_ternary,
                   accuracy_rel,
                   accuracy_norel))
        
    avg_acc_ternary = sum(acc_ternary) / len(acc_ternary)
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    summary_writer.add_scalars('Accuracy/train', {
        'ternary': avg_acc_ternary,
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    avg_loss_ternary = sum(l_ternary) / len(l_ternary)
    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    summary_writer.add_scalars('Loss/train', {
        'ternary': avg_loss_ternary,
        'binary': avg_loss_binary,
        'unary': avg_loss_unary
    }, epoch)

    # return average accuracy
    return avg_acc_ternary, avg_acc_binary, avg_acc_unary

def test(epoch, ternary, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_ternary = []
    accuracy_rels = []
    accuracy_norels = []

    loss_ternary = []
    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)
        acc_ter, l_ter = model.test_(input_obj, input_qst, label)
        accuracy_ternary.append(acc_ter.item())
        loss_ternary.append(l_ter.item())

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_obj, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_obj, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Ternary accuracy: {:.0f}% Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(
        accuracy_ternary, accuracy_rel, accuracy_norel))

    summary_writer.add_scalars('Accuracy/test', {
        'ternary': accuracy_ternary,
        'binary': accuracy_rel,
        'unary': accuracy_norel
    }, epoch)

    loss_ternary = sum(loss_ternary) / len(loss_ternary)
    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/test', {
        'ternary': loss_ternary,
        'binary': loss_binary,
        'unary': loss_unary
    }, epoch)

    return accuracy_ternary, accuracy_rel, accuracy_norel

    
def load_data():
    print('loading data...')
    dirs = './data'
    
    #load dataset to train on based on arguments passed in
    if args.dataset == 'pixels':
        fname = 'sort-of-clevr.pickle'
    else:
        fname = 'sort-of-clevr_state_desc.pickle'

    filename = os.path.join(dirs,fname)

    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    ternary_train = []
    ternary_test = []
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')
    for obj, ternary, relations, norelations in train_datasets:
        if args.dataset == "pixels": obj = np.swapaxes(obj, 0, 2)

        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_train.append((obj,qst,ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((obj,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((obj,qst,ans))

    for obj, ternary, relations, norelations in test_datasets:
        if args.dataset == "pixels": obj = np.swapaxes(obj, 0, 2)
        
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_test.append((obj, qst, ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((obj,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((obj,qst,ans))
    
    return (ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test)
    

ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    print(filename)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

with open(f'./{args.model}_{args.seed}_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_ternary', 'train_acc_rel',
                     'train_acc_norel', 'train_acc_ternary', 'test_acc_rel', 'test_acc_norel'])

    print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")

    for epoch in range(1, args.epochs + 1):
        train_acc_ternary, train_acc_binary, train_acc_unary = train(
            epoch, ternary_train, rel_train, norel_train)
        test_acc_ternary, test_acc_binary, test_acc_unary = test(
            epoch, ternary_test, rel_test, norel_test)

        csv_writer.writerow([epoch, train_acc_ternary, train_acc_binary,
                         train_acc_unary, test_acc_ternary, test_acc_binary, test_acc_unary])
        model.save_model(epoch)
