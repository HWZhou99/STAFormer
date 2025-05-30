from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp

import torch
import time
import math

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

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

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def print_time(string=''):
    ctime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    res = ctime + ' | ' + string
    print(res)

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
#计算帧级损失
def DeepSupervision(criterion , xs, y , mode='CE'):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """

    loss = 0.
    if mode =='CE-frame':
        batch_size = y.size(0)
        for x in xs:
            len_frame = x.size(0) // batch_size
            x = x.reshape(batch_size, len_frame, x.size(1))
            for i in range(len_frame):
                loss = loss+criterion(x[:,i,:], y)
            loss = loss / len_frame
        loss /= len(xs)

    if mode =='Trip-frame':
        batch_size = y.size(0)
        for x in xs:
            len_frame = x.size(0) // batch_size
            x = x.reshape(batch_size, len_frame, x.size(1))
            for i in range(len_frame):
                loss = loss+criterion(x[:, i, :], y)
            loss = loss / len_frame
        loss /= len(xs)
        
    if mode =='Arc-frame':
        batch_size = y.size(0)
        for x in xs:
            len_frame = x.size(0) // batch_size
            x = x.reshape(batch_size, len_frame, x.size(1))
            for i in range(len_frame):
                loss = loss+criterion(x[:, i, :], y)
            loss = loss / len_frame
        loss /= len(xs)
    return loss
        
