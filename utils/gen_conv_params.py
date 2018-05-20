__all__ = ['squaredims', 'cal_padding_size', 'get_iter', 'gen_conv_params', 'get_itemset',
          'join_dict', 'reduce_projections']

import functools

def assert_int_or_list(n):
    assert isinstance(n, int) or (isinstance(n, (list, tuple)) and len(n) > 0), (
    'n should be an int or a non-empty list or tuple, but is {0}'.format(n))
    
def squaredims(n):
    """Given an integer, calculate the factorization n = a * b, where |a - b| <= 2, a <= b
    >>> squaredims(100)
    (10, 10)
    """
    import math
    a = math.floor(math.sqrt(n))
    b = math.ceil(n / a)
    return a, b

def cal_padding_size(a, kernel_size=3, stride=2):
    """Calculate padding size (equal on both sides)
    Args:
        a: int or tuple; input size ()
    >>> cal_padding_size((10), 4)
    0
    >>> cal_padding_size((10,), 4)
    (0,)
    >>> cal_padding_size(10, 4)
    0
    >>> cal_padding_size((10, 8))
    (1, 1)
    """
    assert_int_or_list(a)
    if not isinstance(a, int):
        p = ()
        for i in a:
            p = p + (cal_padding_size(i, kernel_size, stride),) 
        return p
    # do not pad if a < kernel_size (in this case change kernel size)
    if a < kernel_size:
        return 0
    if (a - kernel_size + stride) % stride == 0:
        return 0
    else: 
        p = stride - (a - kernel_size + stride) % stride
        return (p + 1) // 2

def get_iter(k, length=None):
    """Return an iterator for two kinds of input: int, or tuple/list
    Args:
        k: int or tuple/list
        length: int or None; default None. If not None, return the iterator of a list of size length
    >>> list(get_iter([1, 2], 4))
    [1, 2, 2, 2]
    >>> list(get_iter([1], 1))
    [1]
    >>> list(get_iter(1, 3))
    [1, 1, 1]
    >>> list(get_iter(2))
    [2]
    >>> list(get_iter((3, 4, 5)))
    [3, 4, 5]
    """
    assert_int_or_list(k)
    assert length is None or (isinstance(length, int) and length > 0)
    if isinstance(k, (tuple, list)):
        if length is None:
            return iter(k)
        else:
            if len(k) < length:
                k = list(k) + [k[len(k)-1]] * (length - len(k)) # elongate k by repeating its last value
            return iter(k[:length])
    else:
        if length is None:
            return iter([k])
        else:
            return iter([k] * length)

def gen_conv_params(n, kernel_size=3, stride=2, threshold=40):
    """Calculate conv_params;
    Args:
        n: int or tuple/list; e.g., the dimentionality of 1d or 2d features
        kernel_size: int or tuple/list, default 3
        stride: int or tuple/list, default 2
        threshold: int or tuple/list, default 40; if the number of features n is less than threshold, 
        then set kernel_size = n
    Return:
        a list of [kernel_size, stride, padding]
    >>> gen_conv_params(10, threshold=1)
    [[3, 2, 1], [3, 2, 0], [2, 1, 0]]
    >>> gen_conv_params([10, 5], threshold=5)
    [[(3, 3), (2, 2), (1, 0)], [(3, 2), (2, 1), (0, 0)], [(2, 1), (1, 1), (0, 0)]]
    >>> gen_conv_params((10,1), threshold=5)
    [[(3, 1), (2, 1), (1, 0)], [(3, 1), (2, 1), (0, 0)], [(2, 1), (1, 1), (0, 0)]]
    """
    [assert_int_or_list(e) for e in [n, kernel_size, stride, threshold]]
    if isinstance(n, (tuple, list)):
        iter_threshold = get_iter(threshold, len(n))
        res = [gen_conv_params(i, kernel_size, stride, next(iter_threshold)) for i in n]
        max_depth = max([len(e) for e in res])
        return functools.reduce(lambda a, b: [[a[i][0]+(b[i][0],), a[i][1]+(b[i][1],), a[i][2]+(b[i][2],)] 
                                              if i<len(b) else [a[i][0]+(1,), a[i][1]+(1,), a[i][2]+(0,)]
                                              for i in range(max_depth)],
                                res, [[(), (), ()]]*max_depth)
    else:
        if n < threshold:
            return [[n, 1, 0]]
        iter_kernel_size = get_iter(kernel_size)
        kernel_size = next(iter_kernel_size)
        iter_stride = get_iter(stride)
        stride = next(iter_stride)
        conv_params = []
        while n > kernel_size:
            p = cal_padding_size(n, kernel_size, stride)
            k = n if n < kernel_size else kernel_size
            conv_params.append([k, stride, p])
            n = (n + 2 * p - k) // stride + 1
            try:
                kernel_size = next(iter_kernel_size)
            except StopIteration:
                pass
            try:
                stride = next(iter_stride)
            except StopIteration:
                pass
        conv_params.append([n, 1, 0])
    return conv_params

def get_itemset(keys, dic):
    """Merge elements of dic given keys
        The item values of dic should be comparable
    >>> get_itemset([1], {1: ['1', 'a']})
    ['1', 'a']
    """
    return sorted(functools.reduce(lambda x, y: set(x).union(y),
                                   [dic[k] for k in keys]))

def join_dict(dict0, dict1):
    """Join two dictionaries. values (type: list) of dict1 is keys of dict0
    >>> join_dict({1: [2], 2: [3], 3:[5]}, {1:[1], 2:[6]})
    {1: [2]}
    >>> join_dict({}, {})
    {}
    >>> join_dict({'a': ['b', '1']}, {'test': ['a']})
    {'test': ['1', 'b']}
    """
    return {k: get_itemset(set(v).intersection(dict0.keys()), dict0) for k, v in dict1.items() 
            if set(v).intersection(dict0.keys())}

def reduce_projections(projections, del_unused_input=True):
    """Given a ordered list of projections, return reduced version that only includes paticipating nodes 
    Args: 
        (ordered) list of dictionaries (keys are int, values are list of int)
    Return: 
        projections: a list of lists of int. Since now keys are consecutive int starting from 0, 
            representing it using a list instead of dict
        idx_to_var: list of ints, mapping 0-len(proj) to their original index
    """
    import copy
    import collections
    assert len(projections) > 0, 'projections should be a non-empty list of dictionaries'
    assert all([isinstance(proj, (dict, collections.defaultdict)) for proj in projections])
    projections = [copy.deepcopy(proj) for proj in projections]
    # remove possible empty entries in projections[0]
    projections[0] = {k: v for k, v in projections[0].items() if v}
    for i in range(1, len(projections)):
        projections[i] = {k: set(v).intersection(projections[i-1].keys()) 
                          for k, v in projections[i].items()
                         if set(v).intersection(projections[i-1].keys())}
    projections = [proj for proj in projections if proj]
    # Delete unused input so that any node in a lower layer 
    # will "flow" (connect) to the last layer
    if del_unused_input:
        for i in range(1, len(projections)):
            allkeys = get_itemset(projections[i].keys(), projections[i])
            projections[i-1] = {k: v for k, v in projections[i-1].items()
                                if k in allkeys}
    input_to_idx = {k: i for i, k in enumerate(get_itemset(projections[0].keys(), projections[0]))}
    var_to_idx = [{k: i for i, k in enumerate(sorted(proj.keys()))} for proj in projections]
    projections = [{var_to_idx[i][k]: sorted([input_to_idx[v] for v in proj[k]]) if i == 0
                    else sorted([var_to_idx[i-1][v] for v in proj[k]]) 
                    for k in sorted(proj.keys())}
                   for i, proj in enumerate(projections)]
    idx_to_var = [sorted(m.keys()) for m in [input_to_idx] + var_to_idx]
    return projections, idx_to_var

if __name__ == '__main__':
    import doctest
    doctest.testmod()