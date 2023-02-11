from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
sys.path.append('./torchFewShot')

from args_xent import argument_parser
#from args_cifar import argument_parser

from torchFewShot.models.net import Model
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate

from tqdm import tqdm

from lr_helper import warmup_scheduler
from patchmix_utils import *
parser = argument_parser()
args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    model_name = '{}-{}shot-{}'.format(args.dataset, args.nExemplars, args.backbone)

    if args.suffix is not None:
        model_name = model_name + '-{}'.format(args.suffix)

    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    support = True
    model = Model(num_classes=args.num_classes)
    ori_model = Model(num_classes=args.num_classes)
    ori_model.load_state_dict(torch.load(args.resume)['state_dict'], strict=False)
    criterion = CrossEntropyLoss()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
    
    if use_gpu:
        model = model.cuda()
        ori_model = ori_model.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")
    
    warmup_epoch = 5
    scheduler = warmup_scheduler(base_lr=args.lr, iter_per_epoch=len(trainloader), 
    max_epoch=args.max_epoch + warmup_epoch, multi_step=[], warmup_epoch=warmup_epoch)

    for epoch in range(args.max_epoch + warmup_epoch):

        start_train_time = time.time()
        train(epoch, model, ori_model, criterion, optimizer, trainloader, scheduler, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            acc = test(model, testloader, use_gpu)
            is_best = acc > best_acc
            
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            
                save_checkpoint({'state_dict': model.state_dict(),'acc': acc,'epoch': epoch,}, is_best, osp.join(save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
            save_checkpoint({'state_dict': model.state_dict(),'acc': acc,'epoch': epoch,}, is_best, osp.join(save_dir, 'checkpoint_last.pth.tar'))

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, ori_model, criterion, optimizer, trainloader, scheduler, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    ori_model.eval()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids, pids_train) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids_test = pids.cuda()
            pids_train = pids_train.cuda()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)

        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()
        #### patch mix
        images_test, labels_test_patch, pids_test_patch, _, _, _ = patchmix_hard(ori_model, images_train, images_test, labels_train_1hot, labels_test, pids_test)
        ######################

        ytest, cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids.view(-1))
        with torch.no_grad():
            ytest_, cls_scores_ = ori_model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids.view(-1))

        loss1 = patch_loss(ytest, pids_test_patch, pids_train, True)
        loss2 = patch_loss(cls_scores[0], labels_test_patch, labels_train)
        if args.nExemplars == 1:
            loss3 = nn.MSELoss()(ytest, ytest_) + nn.MSELoss()(cls_scores[0], cls_scores_[0])
        else:
            loss3 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(ytest.permute(0, 2, 3, 1).reshape(-1, 64), -1), F.softmax(ytest_.permute(0, 2, 3, 1).reshape(-1, 64), -1)) + nn.KLDivLoss(reduction='batchmean')(F.log_softmax(cls_scores[0].permute(0, 2, 3, 1).reshape(-1, 5), -1), F.softmax(cls_scores_[0].permute(0, 2, 3, 1).reshape(-1, 5), -1))
        loss = loss1 + 0.5 * loss2 + 0.5 * loss3

        optimizer.zero_grad()
        loss.backward()
        learning_rate = scheduler.step(optimizer)
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
           epoch+1, learning_rate, batch_time=batch_time, 
           data_time=data_time, loss=losses))


def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot, return_test=True)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.extend(acc.tolist())

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main()
