# -*- coding: utf-8 -*-


from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tripletnet import Tripletnet
from copy import deepcopy
import numpy as np
import pickle

###
from FeatureExtractorSquare import FeatureExtractor, Inception
from triplet_image_loader_square import TripletImageLoader
###
# Training settings

def main(train_batch_size, test_batch_size, epochs, resume = ''):
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--train-batch-size', type=int, default = train_batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default = test_batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default = epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--resume', default=resume, type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='TripletNet', type=str,
                        help='name of experiment')
    global args, best_acc, device
    best_acc = 0
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    global train_data, test_data
    kwrgs = {'num_workers': 0, 'pin_memory': True}
    
    train_data = TripletImageLoader("train.txt", True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.train_batch_size, shuffle=True, **kwrgs)
    
    test_data = TripletImageLoader("test.txt", True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size, shuffle=True, **kwrgs)
       
    model = FeatureExtractor()
    tnet = Tripletnet(model).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
        torch.cuda.empty_cache()
        # evaluate on validation set
        test(test_loader, tnet, criterion, epoch)
        # remember best acc and save checkpoint
        save_net({'epoch': epoch + 1,
                  'state_dict': tnet.state_dict()},
                  filename = "model_"+str(epoch)+".pth")
        torch.cuda.empty_cache()
    
    
    
def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    #emb_norms = AverageMeter()
    loss_fn = nn.TripletMarginLoss(margin=0.001, p=2.0)

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        global device
        
        data1 = Variable(data1.to(device))
        data2 = Variable(data2.to(device))
        data3 = Variable(data3.to(device))
        
        # compute output
        dist_an, dist_ap, anchor, neg, pos = tnet(data1, data2, data3)
        loss = loss_fn(anchor, pos, neg)
        # measure accuracy and record loss
        acc = accuracy(dist_an, dist_ap)
        losses.update(loss, data1.size(0))
        accs.update(acc, data1.size(0))
       # emb_norms.update(loss_embedd.item()/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'.format(
                epoch, batch_idx * len(data1), train_data.dataset,
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg))


def test(test_loader, tnet, criterion, epoch):
    global device
    losses = AverageMeter()
    accs = AverageMeter()
    loss_fn = nn.TripletMarginLoss(margin=0.001, p=2.0)

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):

        data1 = Variable(data1.to(device))
        data2 = Variable(data2.to(device))
        data3 = Variable(data3.to(device))  
        
        # compute output
        dist_an, dist_ap, _, _, _ = tnet(data1, data2, data3)
        dist_an, dist_ap, anchor, neg, pos = tnet(data1, data2, data3)
        loss = loss_fn(anchor, pos, neg)

        # measure accuracy and record loss
        acc = accuracy(dist_an, dist_ap)
        accs.update(acc, data1.size(0))
        losses.update(loss, data1.size(0)) 
        
        if batch_idx % args.log_interval == 0:
            print('Test Epoch: {} [{}/{}]\t'.format(
                epoch, batch_idx * len(data1), test_data.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(losses.avg,100. * accs.avg))
        
           
def save_net(state, filename="model_ckpt.pth"):
    """
    Saves checkpoint to disk
    """
    directory = "Checkpoints/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, filename)
    
    

def accuracy(dist_an, dist_ap):
    margin = 0
    pred = (dist_an - dist_ap + margin).to(device).data
    return float((pred > 0).sum()*1.0)/dist_an.size()[0]


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main(train_batch_size = 64, 
         test_batch_size = 32,
         epochs = 10)    
