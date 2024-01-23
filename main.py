"""
This is a modified version of a codebase taken from CGCNN:
https://github.com/txie-93/cgcnn

We thank them for the excellent codebase.
"""

import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from train import *

from data import *
from model import PeriodicSetTransformer

parser = argparse.ArgumentParser(description='Periodic Set Transformer')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.0001)')
parser.add_argument('--lr-milestones', default=[75, 150, 200], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[75, 150, 200])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                         help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default '
                              '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')
parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',
                    help='choose an optimizer, SGD or Adam, (default: Adam)')
parser.add_argument('--fea-len', default=128, type=int, metavar='N',
                    help='size of the initial embedding')
parser.add_argument('--num-encoders', default=4, type=int, metavar='N',
                    help='number of encoder layers')
parser.add_argument('--num-decoder', default=1, type=int, metavar='N',
                    help='number of decoder layers')
parser.add_argument('--num-heads', default=2, type=int, metavar='N',
                    help='number of attention heads')

parser.add_argument('--disable-composition', action='store_true',
                    help='Disable atomic composition')
parser.add_argument('--disable-pdd-encoding', action='store_true',
                    help='Disable PDD Encoding')



args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_mae_error = 1e10


def main():
    global args, best_mae_error
    components = []

    if not args.disable_pdd_encoding:
        components.append("pdd")
    if not args.disable_composition:
        components.append("composition")


    dataset = PDDDataNormalized(*args.data_options)

    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                      'Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        #sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # build model
    orig_atom_fea_len = dataset[0][0].shape[-1]
    model = PeriodicSetTransformer(orig_atom_fea_len,
                                   args.fea_len,
                                   num_heads=args.num_heads,
                                   n_encoders=args.num_encoders,
                                   decoder_layers=args.num_decoder,
                                   components=components,
                                   use_cuda=args.cuda)

    if args.cuda:
        model.cuda()

    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, normalizer, cuda=args.cuda)
        mae_error = validate(val_loader, model, criterion, normalizer, cuda=args.cuda)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    res = validate(test_loader, model, criterion, normalizer, test=True)
    with open("./results.txt", "a") as f:
        f.write(str(args.data_options) + " --> " + str(res) + "\n")


if __name__ == '__main__':
    main()
