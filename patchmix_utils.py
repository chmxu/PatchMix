import torch
import numpy as np
from torchvision.utils import make_grid
import random
import torch.nn.functional as F
from python_tsp.exact import solve_tsp_dynamic_programming
from torch import nn


def patch_loss(pred, labels_test_patch, labels_train, global_=False, patch_size=11):
    criterion = torch.nn.CrossEntropyLoss()
    pred =  pred.view( pred.size()[0],  pred.size()[1], -1).transpose(1,2).reshape(-1, pred.size()[1])
    if global_:
        labels_test_patch =  labels_test_patch.view(-1)
        if labels_train is not None:
            labels_train_patch = labels_train.unsqueeze(2).unsqueeze(2).repeat(1, 1, patch_size, patch_size).view(-1)
            labels = torch.cat([labels_train_patch,labels_test_patch.view(-1)], 0)
        else:
            labels = labels_test_patch.view(-1)
    elif labels_train is not None:
        labels_train_patch = labels_train.unsqueeze(2).unsqueeze(2).repeat(1, 1, patch_size, patch_size)
        labels = torch.cat([labels_train_patch,labels_test_patch], 1).view(-1)
    else:
        labels = labels_test_patch.view(-1)
    loss = criterion(pred, labels)
    return loss

def mix_patch_loss(pred, labels_test_patch, lam, global_=False, patch_size=11):
    criterion = torch.nn.CrossEntropyLoss()
    labels_test_patch_1 = labels_test_patch[:, :, :, :, 0]
    labels_test_patch_2 = labels_test_patch[:, :, :, :, 1]
    lam = lam.view(labels_test_patch.shape[0], -1)
    lam = lam.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, patch_size, patch_size).view(-1)
    pred =  pred.view( pred.size()[0],  pred.size()[1], -1).transpose(1,2).reshape(-1, pred.size()[1])
    labels_test_patch_1 =  labels_test_patch_1.repeat(1, 1, patch_size, patch_size).view(-1)
    labels_test_patch_2 =  labels_test_patch_2.repeat(1, 1, patch_size, patch_size).view(-1)
    loss = lam * nn.CrossEntropyLoss(reduction='none')(pred, labels_test_patch_1) + (1-lam) * nn.CrossEntropyLoss(reduction='none')(pred, labels_test_patch_2)
    return loss.mean()

def generate_matrix():
    xd = np.random.randint(1, 2)
    yd = np.random.randint(1, 2)
    index = list(range(11))
    x0 = np.random.choice(index, size=xd, replace=False)
    y0 = np.random.choice(index, size=yd, replace=False)
    return x0, y0


def random_block(x):
    x0, y0 = generate_matrix()
    mask = torch.zeros([1, 1, 11, 11], requires_grad=False) +1
    for i in x0:
        for j in y0:
                mask[:, :, i, j] = 0
    mask = mask.float()
    x = x * mask.cuda()
    return x

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def patchmix(test_img, test_label, global_label):
    test_label_patch = test_label.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11)
    global_label_patch = global_label.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11)

    batch_size = test_img.size()[0]
    for i in range(batch_size):
        test_label_patch_slice = test_label_patch[i]
        global_label_patch_slice = global_label_patch[i]
        input = test_img[i]
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(input.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        test_img[i] = input 

        #### calculate patch label
        bbx1, bby1, bbx2, bby2 = float(bbx1), float(bby1), float(bbx2), float(bby2)
        bbx1, bby1, bbx2, bby2 = round(bbx1* 11.0/84.0), round(bby1* 11.0/84.0), round(bbx2* 11.0/84.0), round(bby2 * 11.0/84.0)
        test_label_patch_slice[:, bbx1:bbx2, bby1:bby2] = test_label_patch_slice[rand_index, bbx1:bbx2, bby1:bby2]
        global_label_patch_slice[:, bbx1:bbx2, bby1:bby2] = global_label_patch_slice[rand_index, bbx1:bbx2, bby1:bby2]
        ### #############
        test_label_patch[i] = test_label_patch_slice 
        global_label_patch[i] = global_label_patch_slice 
     
    return test_img, test_label_patch, global_label_patch

def calc_sim(feat):
    '''
    feat: (n, c, h, w)
    '''
    feat = F.normalize(feat.mean(-1).mean(-1), p=2, dim=1)
    sim = -1 * torch.matmul(feat, feat.permute(1, 0))
    perm, distance = solve_tsp_dynamic_programming(sim.detach().cpu().numpy())
    perm = [perm[perm.index(i)+1] if perm.index(i)<len(perm)-1 else 0 for i in range(len(perm))]
    perm = torch.tensor(perm)
    return perm

def patchmix_hard(model, train_img, test_img, train_label_1hot, test_label, global_label):
    test_label_patch = test_label.unsqueeze(2).unsqueeze(2).repeat(1, 1,11, 11)
    global_label_patch = global_label.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11)

    batch_size = test_img.size()[0]
    cond_img = []
    cond_img_ori = []
    masks = []
    for i in range(batch_size):
        test_label_patch_slice = test_label_patch[i]
        global_label_patch_slice = global_label_patch[i]
        mask = torch.ones_like(test_label_patch_slice).cuda()
        label = test_label_patch[i][:, 0, 0]
        input = test_img[i]
        ytrain = train_label_1hot[i].transpose(0, 1)
        cond_img_ = input.clone()
        cond_img_ori_ = input.clone()

        test_label_patch_slice_ = test_label_patch_slice.clone()
        global_label_patch_slice_ = global_label_patch_slice.clone()
        input_ = input.clone()
        lam = np.random.beta(1, 1)

        with torch.no_grad():
            ftrain = model.base(train_img[i])
            b, c, h, w = ftrain.shape
            num_train = ftrain.shape[0]
            ftrain = ftrain.reshape(num_train, -1) 
            ftrain = torch.matmul(ytrain, ftrain)
            ftrain = ftrain.div(ytrain.sum(dim=1, keepdim=True).expand_as(ftrain))
            ftrain = ftrain.reshape(-1, c, h, w)
            rand_cls = calc_sim(ftrain)

        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_, bby1_, bbx2_, bby2_ = precise(bbx1), precise(bby1), precise(bbx2), precise(bby2)
        mask[:, bbx1_:bbx2_, bby1_:bby2_] = 0
        masks.append(mask)

        for j in range(5):
            ori_idx = torch.nonzero(label==j).view(-1)
            target_idx = torch.nonzero(label==rand_cls[j]).view(-1)
            target_idx = target_idx[torch.randperm(target_idx.shape[0])]

            input_[ori_idx, :, bbx1:bbx2, bby1:bby2] = input[target_idx, :, bbx1:bbx2, bby1:bby2]
            cond_img_[ori_idx, ] = input[target_idx, ]
            cond_img_[ori_idx, :, bbx1:bbx2, bby1:bby2] = input[ori_idx, :, bbx1:bbx2, bby1:bby2]
            cond_img_ori_[ori_idx, ] = input[target_idx, ]
            test_label_patch_slice_[ori_idx, bbx1_:bbx2_, bby1_:bby2_] = test_label_patch_slice[target_idx, bbx1_:bbx2_, bby1_:bby2_]
            global_label_patch_slice_[ori_idx, bbx1_:bbx2_, bby1_:bby2_] = global_label_patch_slice[target_idx, bbx1_:bbx2_, bby1_:bby2_]

        test_img[i] = input_
        test_label_patch[i] = test_label_patch_slice_
        global_label_patch[i] = global_label_patch_slice_
        cond_img.append(cond_img_)
        cond_img_ori.append(cond_img_ori_)

    cond_img = torch.stack(cond_img, 0)
    cond_img_ori = torch.stack(cond_img_ori, 0)

    return test_img, test_label_patch, global_label_patch, cond_img, cond_img_ori, torch.stack(masks, 0)

#### from cross attention
def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


import os
import os.path as osp
import errno
import json
import shutil

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def save_checkpoint(state, is_best=False, fpath='checkpoint.pth.tar'):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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

def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot


def make_log(writer, loss, img, p_img, recons, step):
    writer.add_scalar('L1 Loss', loss, step)
    writer.add_image("origin image", make_grid(img[:10,], normalize=True, scale_each=True), step)
    writer.add_image("patchmix image", make_grid(p_img[:10,], normalize=True, scale_each=True), step)
    writer.add_image("recons image", make_grid(recons[:10,], normalize=True, scale_each=True), step)
