import random
import numpy as np
import collections

try:
    import torch
    from torch.autograd import Variable
except ImportError:
    pass

__all__ = ['BatchSequentialSampler', 'RepeatedBatchSampler', 'balanced_sampler', 'BatchLoader']

if torch.cuda.is_available():
  dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor} #pylint disable=no-member
else:
  dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor} 


class BatchSequentialSampler(object):
    """return a list of batches (same implementation with torch.utils.data.sampler.BatchSampler)
    Args:
        sampler: an iterator, eg: range(100)
        batch_size: int
        drop_last: bool
    Return:
        an iterator, each iter returns a batch of batch_size from sampler 
    """
    def __init__(self, sampler, batch_size=1, drop_last=False):
        self.sampler = sampler    
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

        
class RepeatedBatchSampler(object):
    """Generate num_iter of batches with batch_size
    Args:
        sampler: an iterator, that will be converted to list
        batch_size: int
        num_iter: int, default: None
        shuffle: bool, default: True
        allow_duplicate: bool, default: True
    Return:
        an iterator of length num_iter
    """
    def __init__(self, sampler, batch_size=1, num_iter=None, shuffle=True, allow_duplicate=True, seed=None): 
        assert len(sampler) > 0
        self.sampler = sampler # only for __next__(); uselesss
        if len(sampler) < batch_size and not allow_duplicate:
            batch_size = len(sampler)
        assert batch_size > 0
        self.batch_size = batch_size
        if num_iter is None:
            num_iter = (len(sampler) + batch_size - 1) // batch_size
        assert num_iter > 0
        self.num_iter = num_iter
        num_repeats = (num_iter * batch_size + len(sampler) - 1 ) // len(sampler)
        self.sampler_ext = []
        if seed is not None: # if seed: is buggy (seed=0)
            np.random.seed(seed)
        for i in range(num_repeats):
            if shuffle:
                idx = np.random.permutation(len(sampler))
            else:
                idx = range(len(sampler))
            self.sampler_ext += [sampler[i] for i in idx]
            
    def __iter__(self):
        cnt = 0
        for i in range(self.num_iter):
            yield self.sampler_ext[cnt:(cnt + self.batch_size)]
            cnt += self.batch_size
            
    def __len__(self):
        return self.num_iter
    
    def __next__(self):
        indices = np.random.permutation(len(self.sampler))[:self.batch_size]
        sampler = list(self.sampler)
        batch = [sampler[i] for i in indices]
        return batch

def balanced_sampler(y, batch_size=10, num_iter=None, allow_duplicate=False, 
                     max_redundancy=3, shuffle=True, seed=None):
    """Given class labels y, return a balanced batch sampler, i.e., 
       each class appears the same number of times in each batch
    Args:
        y: list, tuple, or numpy 1-d array
        batch_size: int; how many instances of each class should be included in a batch. 
                    Thus the real batch size = batch_size * num_classes in most cases
        num_iter: number of batches. If None, calculate from y, batch_size, etc.
        allow_duplicate: in case batch_size > the smallest class size, if not allow_duplicate, 
                         reduce batch_size
        max_redundancy: default 3; if num_iter is initially None, 
                the calculated num_iter will be larger than num_iter of a 'traditional' epoch 
                by a factor of num_classes. max_redundancy can reduce this factor
        shuffle: default True. Always shuffle the batches
        seed: if not None, call np.random.seed(seed). For unittest
    Return:
        a numpy array of shape (num_iter, real_batch_size)
    """
    z = collections.defaultdict(list)
    # this is extremely buggy; when y is torch.Tensor, e is different even if they have the same value
    [z[e.item()].append(i) for i, e in enumerate(y)] 
    least_size = min([len(v) for k, v in z.items()])
    if least_size < batch_size and not allow_duplicate:
        batch_size = least_size
    if num_iter is None:
        num_iter = (len(y) + batch_size - 1) // batch_size
        if len(z) > max_redundancy:
            num_iter = (num_iter * max_redundancy + len(z) - 1) // len(z)
    
    bs = [RepeatedBatchSampler(v, batch_size=batch_size, num_iter=num_iter, shuffle=shuffle,
                               allow_duplicate=allow_duplicate, seed=seed)
          for k, v in z.items()]
    bs = [[e for e in s] for s in bs]
    indices = np.array(bs).transpose(1, 0, 2).reshape(num_iter, -1)
    # In each batch, shuffle instances so that instances of the same class won't cluster together
    # may not be necessary
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        [np.random.shuffle(v) for v in indices]
    return indices


class BatchLoader(object):
    """Return an iterator of data batches
    Args:
        data: a single or a list/tuple of np.array/torch.Tensor
        labels: class labels, e.g., a list of int, used for balanced_sampler
        batch_size: int
        balanced: if true, used balanced_sampler, else use BatchSequentialSampler
        The rest of parameters are to be passed to balanced_sampler
    """
    def __init__(self, data, batch_size=10, labels=None, balanced=True, num_iter=None, 
                 allow_duplicate=False, max_redundancy=3, shuffle=True, seed=None):
        assert (labels is None and isinstance(data, (tuple, list)) and len(data) > 1) or (
            labels is not None)
        if labels is None:
            labels = data[-1]
        if not isinstance(data, (tuple, list)):
            data = [data]
        assert len(data) > 0 and len(data[0]) == len(labels)
        self.data = data
        N = len(labels)   
        if balanced:
            self.indices = balanced_sampler(labels, batch_size=batch_size, num_iter=num_iter, 
                                            allow_duplicate=allow_duplicate, max_redundancy=max_redundancy,
                                            shuffle=shuffle, seed=seed)
        else:
            idx = range(N)
            if shuffle:
                idx = np.random.permutation(N).tolist()
            self.indices = RepeatedBatchSampler(idx, batch_size=batch_size, 
                num_iter=num_iter, shuffle=shuffle, allow_duplicate=allow_duplicate, seed=seed)
    
    def __iter__(self):
        for idx in self.indices:
            batch = []
            for data in self.data:
                try:
                    if isinstance(data, torch.Tensor):
                        idx = torch.LongTensor(idx)
                except NameError:
                    pass
                batch.append(data[idx]) 
            yield batch
                
    def __len__(self):
        return len(self.indices)