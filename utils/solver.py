import time
import shutil
import os.path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models

from .utils import AverageMeter, check_acc
from ..models.densenet import DenseNet
from .sampler import BatchLoader

if torch.cuda.is_available():
  dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor} 
else:
  dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor} 


class Solver(object):
    """Solver
    Args:
        model: 
        data:
        optimizer: e.g., torch.optim.Adam(model.parameters())
        loss_fn: loss function; e.g., torch.nn.CrossEntropy()
        resume: file path to checkpoint
    """
    def __init__(self, model, data, optimizer, loss_fn, resume=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # keep track of loss and accuracy during training
        self.losses_train = []
        self.losses_val = []
        self.acc_train = []
        self.acc_val = []
        self.best_acc_val = 0
        self.epoch_counter = 0
        
        if resume:
            if os.path.isfile(resume):
                checkpoint = torch.load(resume)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer = checkpoint['optimizer']
                self.best_acc_val = checkpoint['best_acc_val']
                self.epoch_counter = checkpoint['epoch']
                self.losses_train = checkpoint['losses_train']
                self.losses_val = checkpoint['losses_val']
                self.acc_train = checkpoint['acc_train']
                self.acc_val = checkpoint['acc_val']
            else:
                print("==> No checkpoint found at '{}'".format(resume))
        
    def _reset_avg_meter(self):
        """reset loss_epoch, top1, top5, batch_time at the beginning of each epoch
        """
        self.loss_epoch = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.batch_time = AverageMeter()
        
    
    def run_one_epoch(self, epoch, batch_size=100, num_samples=None, print_every=100, 
                      training=True, balanced_sample=False, topk=5):
        """run one epoch for training or validating
        Args:
            epoch: int; epoch_counter; used for printing only
            batch_size: int, default: 100
            num_samples: int, default: None. 
                How many samples to use in case we don't want train a whole epoch
            print_every: int, default: 100
            training: bool, default:True. If true, train; else validate
            balanced_sample: default: False. Used for unbalanced dataset
        """
        if 'train_loader' in self.data:
            # This is for image related tasks
            dataloader = self.data['train_loader'] if training else self.data['val_loader']
            # This is very important! dataloader.batch_size is controlled by dataloader.batch_sampler.batch_size
            # not the other way around. This is (probably) due to the fact that dataloader was created by setting batch_size
            dataloader.batch_sampler.batch_size = batch_size
            N = len(dataloader.dataset.imgs)
            num_chunks = (N + batch_size - 1) // batch_size
        elif 'X_train' in self.data:
            X, y = (self.data['X_train'], self.data['y_train']) if training else (self.data['X_val'], self.data['y_val'])
            N = X.size(0)
            if num_samples:
                if num_samples < N and num_samples > 0:
                    N = num_samples
                    
            if balanced_sample and isinstance(y, dtype['long']):
                dataloader = BatchLoader((X[:N], y[:N]), batch_size)
                num_chunks = len(dataloader)
            else:
                shuffle_idx = torch.randperm(N)
                X = torch.index_select(X, 0, shuffle_idx)
                y = torch.index_select(y, 0, shuffle_idx)
                num_chunks = (N + batch_size - 1) // batch_size
                X_chunks = X.chunk(num_chunks)
                y_chunks = y.chunk(num_chunks)
                dataloader = zip(X_chunks, y_chunks)
        else:
            raise ValueError('data must contain either X_train or train_loader')
        
        if training:
            print("Training:")
        else:
            print("Validating:")
            
        self._reset_avg_meter()
        end_time = time.time()
        for i, (X, y) in enumerate(dataloader):
            X = Variable(X)
            y = Variable(y)
            
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.loss_epoch.update(loss.item(), y.size(0))
            # For classification tasks, y.data is torch.LongTensor
            # For regression tasks, y.data is torch.FloatTensor
            is_classification = isinstance(y.data, dtype['long'])
            if is_classification:
                res = check_acc(y_pred, y, (1, topk))
                self.top1.update(res[0].item())
                self.top5.update(res[1].item())
            else:
                # top1 is approximately the 'inverse' of loss
                self.top1.update(1. / (loss.item() + 1.), y.size(0))
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            if training:
                self.losses_train.append(self.loss_epoch.avg)
                self.acc_train.append(self.top1.avg)
            else:
                self.losses_val.append(self.loss_epoch.avg)
                self.acc_val.append(self.top1.avg)
                
            if print_every:
                if (i + 1) % print_every == 0:
                    print('Epoch {0}: iteration {1}/{2}\t'
                          'loss: {losses.val:.3f}, avg: {losses.avg:.3f}\t'
                          'Prec@1: {prec1.val:.3f}, avg: {prec1.avg:.3f}\t'
                          'Prec@5: {prec5.val:.3f}, avg: {prec5.avg:.3f}\t'
                          'batch time: {batch_time.val:.3f} avg: {batch_time.avg:.3f}'.format(
                              epoch + 1, i + 1, num_chunks, losses=self.loss_epoch, prec1=self.top1, 
                              prec5=self.top5, batch_time=self.batch_time))
                    sys.stdout.flush()
            
        return self.top1.avg
    
    def train_eval(self, num_iter=100, batch_size=100, X=None, y=None, X_val=None, y_val=None,
                   X_test=None, y_test=None, eval_test=False, balanced_sample=False, allow_duplicate=False,
                   max_redundancy=1000, seed=None):
        if X is None or y is None:
            X, y = self.data['X_train'], self.data['y_train']
        # Currently only for classification tasks, y is torch.LongTensor 
        assert isinstance(y, dtype['long'])
        if X_val is None or y_val is None:
            X_val, y_val = self.data['X_val'], self.data['y_val']
        if eval_test and (X_test is None or y_test is None):
            X_test, y_test = self.data['X_test'], self.data['y_test']
        
        dataloader_train = BatchLoader((X, y), batch_size, balanced=balanced_sample, 
            num_iter=num_iter, allow_duplicate=allow_duplicate, max_redundancy=max_redundancy, 
            shuffle=True, seed=seed)
        dataloader_val = BatchLoader((X_val, y_val), batch_size, balanced=balanced_sample, 
            num_iter=num_iter, allow_duplicate=allow_duplicate, max_redundancy=max_redundancy, 
            shuffle=True, seed=seed)
        if X_test is not None:
            dataloader_test = BatchLoader((X_test, y_test), batch_size, balanced=balanced_sample, 
                num_iter=num_iter, allow_duplicate=allow_duplicate, max_redundancy=max_redundancy, 
                shuffle=True, seed=seed)
        else:
            dataloader_test = [None]*num_iter

        loss_train_meter = AverageMeter()
        loss_train = {'avg':[], 'batch':[]}
        acc_train_meter = AverageMeter()
        acc_train = {'avg':[], 'batch':[]}
        loss_val_meter = AverageMeter()
        loss_val = {'avg':[], 'batch':[]}
        acc_val_meter = AverageMeter()
        acc_val = {'avg':[], 'batch':[]}
        loss_test_meter = AverageMeter()
        loss_test = {'avg':[], 'batch':[]}
        acc_test_meter = AverageMeter()
        acc_test = {'avg':[], 'batch':[]}

        def forward(X, y, loss_meter, losses, acc_meter, acc, training=False):
            X = Variable(X)
            y = Variable(y)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            loss_meter.update(loss.item(), y.size(0))
            losses['avg'].append(loss_meter.avg)
            losses['batch'].append(loss.item())
            res = check_acc(y_pred, y, (1,))
            acc_meter.update(res[0].item(), y.size(0))
            acc['avg'].append(acc_meter.avg)
            acc['batch'].append(res[0].item())

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            return y_pred, loss

        for (X, y), (X_val, y_val), test_data in zip(dataloader_train, 
                dataloader_val, dataloader_test):        
            forward(X, y, loss_train_meter, loss_train, acc_train_meter, acc_train, 
                training=True)
            forward(X_val, y_val, loss_val_meter, loss_val, acc_val_meter, acc_val, 
                training=False)
            if test_data is not None:
                X_test, y_test = test_data
                forward(X_test, y_test, loss_test_meter, loss_test, acc_test_meter, 
                    acc_test, training=False)
        
        if eval_test:
            return loss_train, acc_train, loss_val, acc_val, loss_test, acc_test
        else:
            return loss_train, acc_train, loss_val, acc_val

    
    def train(self, num_epoch = 10, batch_size=100, num_samples=None, print_every=100, 
              use_validation = True, save_checkpoint=True, file_prefix='', balanced_sample=False, topk=5):
        """train
        Args:
            num_epoch: int, default: 100 
            batch_size: int, default: 100
            num_samples: int, default: None
            print_every: int, default: 100
            use_validation: bool, default: True. If True, run_one_epoch for both training and validating
            save_checkpoint: bool, default: True. If True, save checkpoint with name (file_prefix + 'checkpoint%d.pth' % self.epoch_counter) and best model (file_prefix + 'model_best.pth').
            file_prefix: str, default:''
            balanced_sample: bool; used for sampling balanced batches from unbalanced dataset
        """
        for i in range(self.epoch_counter, self.epoch_counter + num_epoch):
            accuracy = self.run_one_epoch(i, batch_size, num_samples, print_every,
                                          balanced_sample=balanced_sample, topk=topk)
            # In case we don't want validation set. Very rare
            if use_validation:
                accuracy = self.run_one_epoch(i, batch_size, num_samples, print_every, 
                                              training=False, balanced_sample=balanced_sample, topk=topk)
            
            if accuracy > self.best_acc_val:
                self.best_acc_val = accuracy
                if save_checkpoint:
                    state = {'model_state': self.model.state_dict(), 
                            'optimizer': self.optimizer,
                            'best_acc_val': self.best_acc_val,
                            'epoch': i + 1,
                            'losses_train': self.losses_train,
                            'losses_val': self.losses_val,
                            'acc_train': self.acc_train,
                            'acc_val': self.acc_val}
                    filename = file_prefix + 'checkpoint%d.pth' % (i + 1)
                    torch.save(state, filename)
                    shutil.copyfile(filename, file_prefix + 'model_best.pth')
    
    def predict(self, batch_size=100, save_file=True, file_prefix='', X=None, y=None, topk=5, verbose=False):
        """predict
        Args:
            batch_size: int, default: 100; can be larger for large memory
            save_file: bool, default: True; if true, save file
            file_prefix: save file name: file_prefix + 'y_test.pth'
            X: default: None. If not None, use X instead of self.data['X_test']
            y: default: None. Similary to X
        """
        if X is None:
            if 'X_test' in self.data:
                X = self.data['X_test']
            elif 'test_loader' in self.data:
                X = self.data['test_loader']
                dataloader = X
            else:
                raise ValueError('If X is None, then self.data '
                                 'must contain either X_test or test_loader')
            
        if y is None and 'y_test' in self.data:
                y = self.data['y_test']
        
        is_truth_avail = isinstance(y, dtype['long']) or isinstance(y, dtype['float'])
        
        if isinstance(X, dtype['float']):
            N = X.size(0)
            num_chunks = (N + batch_size - 1) // batch_size
            X_chunks = X.chunk(num_chunks)
            dataloader = X_chunks
        
        if is_truth_avail:
            N = y.size(0)
            num_chunks = (N + batch_size - 1) // batch_size
            y_chunks = y.chunk(num_chunks)
        else:
            y_chunks = [None] * num_chunks
        
        self._reset_avg_meter()
        end_time = time.time()
        y_pred = []
        for X, y in zip(X_chunks, y_chunks):  
            X = Variable(X)
            y = Variable(y)
            
            y_pred_tmp = self.model(X) # sometimes model output a tuple
            
            if is_truth_avail:
                loss = self.loss_fn(y_pred_tmp, y)
                self.loss_epoch.update(loss.item(), y.size(0))
                if isinstance(y.data, dtype['long']):
                    res = check_acc(y_pred_tmp, y, (1, topk))
                    self.top1.update(res[0].item())
                    self.top5.update(res[1].item())
                else:
                    self.top1.update(1. / (loss.item() + 1.), y.size(0))
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()
            if isinstance(y_pred_tmp, tuple):
                y_pred_tmp = y_pred_tmp[0]
            y_pred.append(y_pred_tmp)
        
        if is_truth_avail and verbose:
            print('Test set: loss: {losses.avg:.3f}\t'
                  'AP@1: {prec1.avg:.3f}\t'
                  'AP@5: {prec5.avg:.3f}\t'
                  'batch time: {batch_time.avg:.3f}'.format(
                      losses=self.loss_epoch, prec1=self.top1, 
                      prec5=self.top5, batch_time=self.batch_time))
            sys.stdout.flush()
        y_pred = torch.cat(y_pred, 0)
        if save_file:
            torch.save({'y_pred': y_pred}, file_prefix + 'y_pred.pth')
        return y_pred


if __name__ == '__main__':

    mnist_train = torchvision.datasets.MNIST('/projects/academic/jamesjar/tianlema/dl-datasets/mnist',
                                             transform=transforms.Compose([transforms.ToTensor(), 
                                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=200)

    mnist_test = torchvision.datasets.MNIST('/projects/academic/jamesjar/tianlema/dl-datasets/mnist',
                                            transform=transforms.Compose([transforms.ToTensor(), 
                                                                          transforms.Normalize((0.1307,), (0.3081,))]), 
                                             train=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=200)

    train = list(train_loader)
    train = list(zip(*train))
    X_train = torch.cat(train[0], 0)
    y_train = torch.cat(train[1], 0)

    X_val = X_train[50000:]
    y_val = y_train[50000:]
    X_train = X_train[:50000]
    y_train = y_train[:50000]

    test = list(test_loader)
    test = list(zip(*test))
    X_test = torch.cat(test[0], 0)
    y_test = torch.cat(test[1], 0)

    data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 
            'X_test': X_test, 'y_test': y_test}



    
    model = DenseNet(input_param=(1, 64), block_layers=(6, 4), num_classes=10, 
                     growth_rate=32, bn_size=2, dropout_rate=0, transition_pool_param=(3, 1, 1))



    loss_fn = nn.CrossEntropyLoss()



    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)


    
    solver = Solver(model, data, optimizer, loss_fn)
    solver.train(num_epoch=2, file_prefix='mnist-')
    solver.predict(file_prefix='mnist-')
