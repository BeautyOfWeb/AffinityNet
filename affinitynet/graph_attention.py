import functools
import collections
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from ..models.transformer import *

if torch.cuda.is_available():
    dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor} 
else:
    dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor} 


def get_iterator(x, n, forced=False):
    r"""If x is int, copy it to a list of length n
    Cannot handle a special case when the input is an iterable and len(x) = n, 
    but we still need to copy it to a list of length n
    """
    if forced:
        return [x] * n
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x] * n
    # Note: np.array, list are always iterable
    if len(x) != n:
        x = [x] * n
    return x

def get_partial_model(model_part, model):
    pretrained_state_dict = {k: v for k, v in model.state_dict().items() if k in model_part.state_dict()}
    state_dict = model_part.state_dict()
    state_dict.update(pretrained_state_dict)
    model_part.load_state_dict(state_dict)

    
class DenseLinear(nn.Module):
    r"""Multiple linear layers densely connected
    
    Args:
        in_dim: int, number of features
        hidden_dim: iterable of int
        nonlinearity: default nn.ReLU()
                        can be changed to other nonlinear activations
        last_nonlinearity: if True, apply nonlinearity to the last output; default False
        dense: if dense, concatenate all previous intermediate features to current input
        forward_input: should the original input be concatenated to current input used when dense is True
                        if return_all is True and return_layers is None and forward_input is True, 
                            then concatenate input with all hidden outputs as final output
        return_all: if True return all layers
        return_layers: selected layers to output; used only when return_all is True
        bias: if True, use bias in nn.Linear()
        
    Shape:
    
    Attributes:
        A series on weight and bias 
    
    Examples:
    
    >>> m = DenseLinear(3, [3,4], return_all=True)
    >>> x = Variable(torch.randn(4,3))
    >>> m(x)
    """
    def __init__(self, in_dim, hidden_dim, nonlinearity=nn.ReLU(), last_nonlinearity=False, dense=True,
                forward_input=False, return_all=False, return_layers=None, bias=True):
        super(DenseLinear, self).__init__()
        num_layers = len(hidden_dim)
        nonlinearity = get_iterator(nonlinearity, num_layers)
        bias = get_iterator(bias, num_layers)
        self.forward_input = forward_input
        self.return_all = return_all
        self.return_layers = return_layers
        self.dense = dense
        self.last_nonlinearity = last_nonlinearity
        
        self.layers = nn.Sequential()
        cnt_dim = in_dim if forward_input else 0
        for i, h in enumerate(hidden_dim):
            self.layers.add_module('linear'+str(i), nn.Linear(in_dim, h, bias[i]))
            if i < num_layers-1 or last_nonlinearity:
                self.layers.add_module('activation'+str(i), nonlinearity[i])
            cnt_dim += h
            in_dim = cnt_dim if dense else h
            
    def forward(self, x):
        if self.forward_input:
            y = [x]
        else:
            y = []
        out = x
        for n, m in self.layers._modules.items():
            out = m(out)
            if n.startswith('activation'):
                y.append(out)
                if self.dense:
                    out = torch.cat(y, dim=-1)
        if self.return_all:
            if not self.last_nonlinearity: # add last output even if there is no nonlinearity
                y.append(out)
            if self.return_layers is not None:
                return_layers = [i%len(y) for i in self.return_layers]
                y = [h for i, h in enumerate(y) if i in return_layers]
            return torch.cat(y, dim=-1)
        else:
            return out
        
    
class FineTuneModel(nn.Module):
    r"""Finetune the last layer(s) (usually newly added) with a pretained model to learn a representation
    
    Args:
        pretained_model: nn.Module, pretrained module
        new_layer: nn.Module, newly added layer
        freeze_pretrained: if True, set requires_grad=False for pretrained_model parameters
        
    Shape:
        - Input: (N, *)
        - Output: 
        
    Attributes:
        All model parameters of pretrained_model and new_layer
    
    Examples:
    
        >>> m = nn.Linear(2,3)
        >>> model = FineTuneModel(m, nn.Linear(3,2))
        >>> x = Variable(torch.ones(1,2))
        >>> print(m(x))
        >>> print(model(x))
        >>> print(FeatureExtractor(model, [0,1])(x))
    """
    def __init__(self, pretrained_model, new_layer, freeze_pretrained=True):
        super(FineTuneModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.new_layer = new_layer
        if freeze_pretrained:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        return self.new_layer(self.pretrained_model(x))
    
    
class FeatureExtractor(nn.Module):
    r"""Extract features from different layers of the model
    
    Args:
        model: nn.Module, the model
        selected_layers: an iterable of int or 'string' (as module name), selected layers
        
    Shape:
        - Input: (N,*)
        - Output: a list of Variables, depending on model and selected_layers
        
    Attributes: 
        None learnable
       
    Examples:
    
        >>> m = nn.Sequential(nn.Linear(2,2), nn.Linear(2,3))
        >>> m = FeatureExtractor(m, [0,1])
        >>> x = Variable(torch.randn(1, 2))
        >>> m(x)
    """
    def __init__(self, model, selected_layers=None, return_list=False):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.selected_layers = selected_layers
        if self.selected_layers is None:
            self.selected_layers = range(len(model._modules))
        self.return_list = return_list
    
    def set_selected_layers(self, selected_layers):
        self.selected_layers = selected_layers
        
    def forward(self, x):
        out = []
        for i, (name, m) in enumerate(self.model._modules.items()):
            x = m(x)
            if i in self.selected_layers or name in self.selected_layers:
                out.append(x)
        if self.return_list:
            return out
        else:
            return torch.cat(out, dim=-1)
    

class WeightedFeature(nn.Module):
    r"""Transform features into weighted features
    
    Args:
        num_features: int
        reduce: if True, return weighted mean
        
    Shape: 
        - Input: (N, *, num_features) where * means any number of dimensions
        - Output: (N, *, num_features) if reduce is False (default) else (N, *)
        
    Attributes:
        weight: (num_features)
        
    Examples::
    
        >>> m = WeightedFeature(10)
        >>> x = torch.autograd.Variable(torch.randn(5,10))
        >>> out = m(x)
        >>> print(out)
    """
    def __init__(self, num_features, reduce=False, magnitude=None):
        super(WeightedFeature, self).__init__()
        self.reduce = reduce
        self.weight = nn.Parameter(torch.empty(num_features))
        # initialize with uniform weight
        self.weight.data.fill_(1)
        self.magnitude = 1 if magnitude is None else magnitude
    
    def forward(self, x):
        self.normalized_weight = torch.nn.functional.softmax(self.weight, dim=0)
        # assert x.shape[-1] == self.normalized_weight.shape[0]
        out = x * self.normalized_weight * self.magnitude
        if self.reduce:
            return out.sum(-1)
        else:
            return out

        
class WeightedView(nn.Module):
    r"""Calculate weighted view
    
    Args:
        num_groups: int, number of groups (views)
        reduce_dimension: bool, default False. If True, reduce dimension dim
        dim: default -1. Only used when reduce_dimension is True
        
    Shape: 
        - Input: if dim is None, (N, num_features*num_groups)
        - Output: (N, num_features)
        
    Attributes:
        weight: (num_groups)
        
    Examples:
    
        >>> model = WeightedView(3)
        >>> x = Variable(torch.randn(1, 6))
        >>> print(model(x))
        >>> model = WeightedView(3, True, 1)
        >>> model(x.view(1,3,2))
    """
    def __init__(self, num_groups, reduce_dimension=False, dim=-1):
        super(WeightedView, self).__init__()
        self.num_groups = num_groups
        self.reduce_dimension = reduce_dimension
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(num_groups))
        self.weight.data.uniform_(-1./num_groups, 1./num_groups)
    
    def forward(self, x):
        self.normalized_weight = nn.functional.softmax(self.weight, dim=0)
        if self.reduce_dimension:
            assert x.size(self.dim) == self.num_groups
            dim = self.dim if self.dim>=0 else self.dim+x.dim()
            if dim == x.dim()-1:
                out = (x * self.weight).sum(-1)
            else:
                # this is tricky for the case when x.dim()>3
                out = torch.transpose((x.transpose(dim,-1)*self.normalized_weight).sum(-1), dim, -1)
        else:
            assert x.dim() == 2
            num_features = x.size(-1) // self.num_groups
            out = (x.view(-1, self.num_groups, num_features).transpose(1, -1)*self.normalized_weight).sum(-1)
        return out    


class GraphAttentionLayer(nn.Module):
    r"""Attention layer
    
    Args:
        in_dim: int, dimension of input
        out_dim: int, dimension of output
        out_indices: torch.LongTensor, the indices of nodes whose representations are 
                     to be computed
                     Default None, calculate all node representations
                     If not None, need to reset it every time model is run
        feature_subset: torch.LongTensor. Default None, use all features
        kernel: 'affine' (default), use affine function to calculate attention 
                'gaussian', use weighted Gaussian kernel to calculate attention
        k: int, number of nearest-neighbors used for calculate node representation
           Default None, use all nodes
        graph: a list of torch.LongTensor, corresponding to the nearest neighbors of nodes 
               whose representations are to be computed
               Make sure graph and out_indices are aligned properly
        use_previous_graph: only used when graph is None
                            if True, to calculate graph use input
                            otherwise, use newly transformed output
        nonlinearity_1: nn.Module, non-linear activations followed by linear layer 
        nonlinearity_2: nn.Module, non-linear activations followed after attention operation
    
    Shape:
        - Input: (N, in_dim) graph node representations
        - Output: (N, out_dim) if out_indices is None 
                  else (len(out_indices), out_dim)
        
    Attributes:
        weight: (out_dim, in_dim)
        a: out_dim if kernel is 'gaussian' 
           out_dim*2 if kernel is 'affine'
           
    Examples:
    
        >>> m = GraphAttentionLayer(2,2,feature_subset=torch.LongTensor([0,1]), 
                        graph=torch.LongTensor([[0,5,1], [3,4,6]]), out_indices=[0,1], 
                        kernel='gaussian', nonlinearity_1=None, nonlinearity_2=None)
        >>> x = Variable(torch.randn(10,3))
        >>> m(x)
    """
    def __init__(self, in_dim, out_dim, k=None, graph=None, out_indices=None, 
                 feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                 nonlinearity_2=None, use_previous_graph=True, reset_graph_every_forward=False,
                no_feature_transformation=False, rescale=True, layer_norm=False, layer_magnitude=100,
                key_dim=None, feature_selection_only=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.graph = graph
        if graph is None:
            self.cal_graph = True
        else:
            self.cal_graph = False
        self.use_previous_graph = use_previous_graph
        self.reset_graph_every_forward = reset_graph_every_forward
        self.no_feature_transformation = no_feature_transformation
        if self.no_feature_transformation:
            assert in_dim == out_dim
        else:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            # initialize parameters
            std = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-std, std)
        self.rescale = rescale
        self.k = k
        self.out_indices = out_indices
        self.feature_subset = feature_subset
        self.kernel = kernel
        self.nonlinearity_1 = nonlinearity_1
        self.nonlinearity_2 = nonlinearity_2
        self.layer_norm = layer_norm
        self.layer_magnitude = layer_magnitude
        self.feature_selection_only = feature_selection_only

        if kernel=='affine':
            self.a = nn.Parameter(torch.Tensor(out_dim*2))
        elif kernel=='gaussian' or kernel=='inner-product' or kernel=='avg_pool' or kernel=='cosine':
            self.a = nn.Parameter(torch.Tensor(out_dim))
        elif kernel=='key-value':
            if key_dim is None:
                self.key = None
                key_dim = out_dim
            else:
                if self.use_previous_graph:
                    self.key = nn.Linear(in_dim, key_dim)
                else:
                    self.key = nn.Linear(out_dim, key_dim)
            self.key_dim = key_dim
            self.a = nn.Parameter(torch.Tensor(out_dim))
        else:
            raise ValueError('kernel {0} is not supported'.format(kernel))
        self.a.data.uniform_(0, 1)
    
    def reset_graph(self, graph=None):
        self.graph = graph
        self.cal_graph = True if self.graph is None else False
        
    def reset_out_indices(self, out_indices=None):
        self.out_indices = out_indices
    
    def forward(self, x):
        if self.reset_graph_every_forward:
            self.reset_graph()
            
        N = x.size(0)
        out_indices = dtype['long'](range(N)) if self.out_indices is None else self.out_indices
        if self.feature_subset is not None:
            x = x[:, self.feature_subset]
        assert self.in_dim == x.size(1)
         
        if self.no_feature_transformation:
            out = x
        else:
            out = nn.functional.linear(x, self.weight)
        
        feature_weight = nn.functional.softmax(self.a, dim=0) 
        if self.rescale and self.kernel!='affine':
            out = out*feature_weight
            if self.feature_selection_only:
                return out

        if self.nonlinearity_1 is not None:
            out = self.nonlinearity_1(out)
        k = N if self.k is None else min(self.k, out.size(0))

        if self.kernel=='key-value':
            if self.key is None:
                keys = x if self.use_previous_graph else out
            else:
                keys = self.key(x) if self.use_previous_graph else self.key(out)
            norm = torch.norm(keys, p=2, dim=-1)
            att = (keys[out_indices].unsqueeze(-2) * keys.unsqueeze(-3)).sum(-1) / (norm[out_indices].unsqueeze(-1)*norm)
            att_, idx = att.topk(k, -1)
            a = Variable(torch.zeros(att.size()).fill_(float('-inf')).type(dtype['float']))
            a.scatter_(-1, idx, att_)
            a = nn.functional.softmax(a, dim=-1)
            y = (a.unsqueeze(-1)*out.unsqueeze(-3)).sum(-2)
            if self.nonlinearity_2 is not None:
                y = self.nonlinearity_2(y)
            if self.layer_norm:
                y = nn.functional.relu(y)  # maybe redundant; just play safe
                y = y / y.sum(-1, keepdim=True) * self.layer_magnitude # <UncheckAssumption> y.sum(-1) > 0
            return y

        # The following line is BUG: self.graph won't update after the first update
        # if self.graph is None
        # replaced with the following line
        if self.cal_graph:
            if self.kernel != 'key-value':
                features = x if self.use_previous_graph else out
                dist = torch.norm(features.unsqueeze(1)-features.unsqueeze(0), p=2, dim=-1)
                _, self.graph = dist.sort()
                self.graph = self.graph[out_indices]               
        y = Variable(torch.zeros(len(out_indices), out.size(1)).type(dtype['float']))
        
        for i, idx in enumerate(out_indices):
            neighbor_idx = self.graph[i][:k]
            if self.kernel == 'gaussian':
                if self.rescale: # out has already been rescaled
                    a = -torch.sum((out[idx] - out[neighbor_idx])**2, dim=1)
                else:
                    a = -torch.sum((feature_weight*(out[idx] - out[neighbor_idx]))**2, dim=1)
            elif self.kernel == 'inner-product':
                if self.rescale: # out has already been rescaled
                    a = torch.sum(out[idx]*out[neighbor_idx], dim=1)
                else:
                    a = torch.sum(feature_weight*(out[idx]*out[neighbor_idx]), dim=1)
            elif self.kernel == 'cosine':
                if self.rescale: # out has already been rescaled
                    norm = torch.norm(out[idx]) * torch.norm(out[neighbor_idx], p=2, dim=-1)
                    a = torch.sum(out[idx]*out[neighbor_idx], dim=1) / norm
                else:
                    norm = torch.norm(feature_weight*out[idx]) * torch.norm(feature_weight*out[neighbor_idx], p=2, dim=-1)
                    a = torch.sum(feature_weight*(out[idx]*out[neighbor_idx]), dim=1) / norm
            elif self.kernel == 'affine':
                a = torch.mv(torch.cat([(out[idx].unsqueeze(0) 
                                         * Variable(torch.ones(len(neighbor_idx)).unsqueeze(1)).type(dtype['float'])), 
                                        out[neighbor_idx]], dim=1), self.a)
            elif self.kernel == 'avg_pool':
                a = Variable(torch.ones(len(neighbor_idx)).type(dtype['float']))
            a = nn.functional.softmax(a, dim=0)
            # since sum(a)=1, the following line should torch.sum instead of torch.mean
            y[i] = torch.sum(out[neighbor_idx]*a.unsqueeze(1), dim=0)
        if self.nonlinearity_2 is not None:
            y = self.nonlinearity_2(y)
        if self.layer_norm:
            y = nn.functional.relu(y)  # maybe redundant; just play safe
            y = y / y.sum(-1, keepdim=True) * self.layer_magnitude # <UncheckAssumption> y.sum(-1) > 0
        return y
        
        
class GraphAttentionModel(nn.Module):
    r"""Consist of multiple GraphAttentionLayer
    
    Args:
        in_dim: int, num_features
        hidden_dims: an iterable of int, len(hidden_dims) is number of layers
        ks: an iterable of int, k for GraphAttentionLayer. 
            Default None, use all neighbors for all GraphAttentionLayer
        kernels, graphs, nonlinearities_1, nonlinearities_2, feature_subsets, out_indices, use_previous_graphs: 
            an iterable of * for GraphAttentionLayer
        
    Shape:
        - Input: (N, in_dim)
        - Output: (x, hidden_dims[-1]), x=N if out_indices is None. Otherwise determined by out_indices
    
    Attributes:
        weights: a list of weight for GraphAttentionLayer
        a: a list of a for GraphAttentionLayer
    
    Examples:
    
        >>> m=GraphAttentionModel(5, [3,4], [3,3])
        >>> x = Variable(torch.randn(10,5))
        >>> m(x)
    """
    def __init__(self, in_dim, hidden_dims, ks=None, graphs=None, out_indices=None, feature_subsets=None,
                 kernels='affine', nonlinearities_1=nn.Hardtanh(), nonlinearities_2=None,
                 use_previous_graphs=True, reset_graph_every_forward=False, no_feature_transformation=False,
                rescale=True):
        super(GraphAttentionModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        num_layers = len(hidden_dims)
        self.no_feature_transformation = get_iterator(no_feature_transformation, num_layers)
        for i in range(num_layers):
            if self.no_feature_transformation[i]:
                if i == 0:
                    assert hidden_dims[0] == in_dim
                else:
                    assert hidden_dims[i-1] == hidden_dims[i]
                
        if ks is None or isinstance(ks, int):
            ks = [ks]*num_layers
        self.ks = ks
        if graphs is None:
            graphs = [None]*num_layers
        self.graphs = graphs
        self.reset_graph_every_forward = reset_graph_every_forward
        if isinstance(kernels, str):
            kernels = [kernels]*num_layers
        self.kernels = kernels
        if isinstance(nonlinearities_1, nn.Module) or nonlinearities_1 is None:
            nonlinearities_1 = [nonlinearities_1]*num_layers
        # Tricky: if nonlinearities_1 is an instance of nn.Module, then nonlinearities_1 will become a 
        # child module of self. Reassignment will have to be a nn.Module
        self.nonlinearities_1 = nonlinearities_1
        if isinstance(nonlinearities_2, nn.Module) or nonlinearities_2 is None:
            nonlinearities_2 = [nonlinearities_2]*num_layers
        self.nonlinearities_2 = nonlinearities_2
        self.out_indices = out_indices
        if isinstance(out_indices, dtype['long']) or out_indices is None:
            self.out_indices = [out_indices]*num_layers
        self.feature_subsets = feature_subsets
        if isinstance(feature_subsets, dtype['long']) or feature_subsets is None:
            self.feature_subsets = [feature_subsets]*num_layers
        self.use_previous_graphs = use_previous_graphs
        if isinstance(use_previous_graphs, bool):
            self.use_previous_graphs = [use_previous_graphs]*num_layers
        self.rescale = get_iterator(rescale, num_layers)
            
        self.attention = nn.Sequential()
        for i in range(num_layers):
            self.attention.add_module('layer'+str(i), 
                GraphAttentionLayer(in_dim if i==0 else hidden_dims[i-1], out_dim=hidden_dims[i], 
                                    k=self.ks[i], graph=self.graphs[i], out_indices=self.out_indices[i],
                                    feature_subset=self.feature_subsets[i], kernel=self.kernels[i],
                                    nonlinearity_1=self.nonlinearities_1[i],
                                    nonlinearity_2=self.nonlinearities_2[i], 
                                    use_previous_graph=self.use_previous_graphs[i],
                                   no_feature_transformation=self.no_feature_transformation[i],
                                   rescale=self.rescale[i]))
            
    def reset_graph(self, graph=None):
        num_layers = len(self.hidden_dims)
        graph = get_iterator(graph, num_layers)
        for i in range(num_layers):
            getattr(self.attention, 'layer'+str(i)).reset_graph(graph[i])
        self.graphs = graph  
            
    def reset_out_indices(self, out_indices=None):
        num_layers = len(self.hidden_dims)
        out_indices = get_iterator(out_indices, num_layers)
        assert len(out_indices) == num_layers
        for i in range(num_layers):
            # probably out_indices should not be a list;
            # only the last layer will output certain points, all previous ones should output all points
            getattr(self.attention, 'layer'+str(i)).reset_out_indices(out_indices[i])
        self.out_indices = out_indices
            # functools.reduce(lambda m, a: getattr(m, a), ('attention.layer'+str(i)).split('.'), self).reset_out_indices(out_indices[i])
        
    def forward(self, x):
        if self.reset_graph_every_forward:
            self.reset_graph()
            
        return self.attention(x)
    
    
class GraphAttentionGroup(nn.Module):
    r"""Combine different view of data
    
    Args:
        group_index: an iterable of torch.LongTensor or other type that can be subscripted by torch.Tensor;
                     each element is feed to GraphAttentionModel as feature_subset
        merge: if True, aggregate the output of each group (view);
               Otherwise, concatenate the output of each group
        in_dim: only used when group_index is None, otherwise determined by group_index
        feature_subset: not used when group_index is not None: always set to None internally
        out_dim, k, graph, out_indices, kernel, nonlinearity_1, nonlinearity_2, and
            use_previous_graph are used similarly in GraphAttentionLayer
            
    Shape:
        - Input: (N, in_dim)
        - Output: (x, y) where x=N if out_indices is None len(out_indices)
                              y=out_dim if merge is True else out_dim*len(group_index)
                              
    Attributes:
        weight: (out_dim, in_dim) 
        a: (out_dim) if kernel='gaussian' else (out_dim * 2)
        group_weight: (len(group_index)) if merge is True else None
        
    Examples:
    
        >>> m = GraphAttentionGroup(2, 2, k=None, graph=None, out_indices=None, 
                 feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                 nonlinearity_2=None, use_previous_graph=True, group_index=[range(2), range(2,4)], merge=False)
        >>> x = Variable(torch.randn(5, 4))
        >>> m(x)
    """
    def __init__(self, in_dim, out_dim, k=None, graph=None, out_indices=None, 
                 feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                 nonlinearity_2=None, use_previous_graph=True, group_index=None, merge=True,
                 merge_type='sum', reset_graph_every_forward=False, no_feature_transformation=False,
                rescale=True, merge_dim=None, layer_norm=False, layer_magnitude=100, key_dim=None):
        super(GraphAttentionGroup, self).__init__()
        self.group_index = group_index
        num_groups = 0 if self.group_index is None else len(group_index) 
        self.num_groups = num_groups
        self.merge = merge
        assert merge_type=='sum' or merge_type=='affine'
        self.merge_type = merge_type
        
        self.components = nn.ModuleList()
        self.group_weight = None
        self.feature_weight = None
        if group_index is None or len(group_index)==1:
            self.components.append(GraphAttentionLayer(in_dim, out_dim, k, graph, out_indices, feature_subset,
                                                       kernel, nonlinearity_1, nonlinearity_2,
                                                       use_previous_graph,
                                                       reset_graph_every_forward=False,
                                                       no_feature_transformation=no_feature_transformation, 
                                                       rescale=rescale, layer_norm=layer_norm,
                                                       layer_magnitude=layer_magnitude, key_dim=key_dim))
        else:
            self.out_dim = get_iterator(out_dim, num_groups)
            self.k = get_iterator(k, num_groups)
            # BUG here: did not handle a special case where len(graph) = num_groups
            self.graph = get_iterator(graph, num_groups)
            # all groups' output have the same first dimention
            self.out_indices = out_indices
            # each group use all of its own features
            self.feature_subset = None
            self.kernel = get_iterator(kernel, num_groups, isinstance(kernel, str))
            self.nonlinearity_1 = get_iterator(nonlinearity_1, num_groups)
            self.nonlinearity_2 = get_iterator(nonlinearity_2, num_groups)
            self.use_previous_graph = get_iterator(use_previous_graph, num_groups)
            self.layer_norm = get_iterator(layer_norm, num_groups)
            self.layer_magnitude = get_iterator(layer_magnitude, num_groups)
            self.key_dim = get_iterator(key_dim, num_groups)
            for i, idx in enumerate(group_index):
                self.components.append(
                    GraphAttentionLayer(len(idx), self.out_dim[i], self.k[i], self.graph[i],
                                        self.out_indices, self.feature_subset, self.kernel[i],
                                        self.nonlinearity_1[i], self.nonlinearity_2[i], 
                                        self.use_previous_graph[i],
                                        reset_graph_every_forward=False,
                                        no_feature_transformation=no_feature_transformation,
                                        rescale=rescale, layer_norm=self.layer_norm[i],
                                        layer_magnitude=self.layer_magnitude[i],
                                        key_dim=self.key_dim[i]))
            if self.merge:
                self.merge_dim = merge_dim if isinstance(merge_dim, int) else self.out_dim[0]
                if self.merge_type=='sum':
                    # all groups' output should have the same dimension
                    for i in self.out_dim:
                        assert i==self.merge_dim
                    self.group_weight = nn.Parameter(torch.Tensor(num_groups))
                    self.group_weight.data.uniform_(-1/num_groups,1/num_groups)
                elif self.merge_type=='affine':
                    # This is ugly and buggy
                    # Do not assume each view have the same out_dim, finally output merge_dim
                    # if merge_dim is None then set merge_dim=self.out_dim[0]
                    self.feature_weight = nn.Parameter(torch.Tensor(self.merge_dim, sum(self.out_dim)))
                    self.feature_weight.data.uniform_(-1./sum(self.out_dim), 1./sum(self.out_dim))
                
    def reset_graph(self, graph=None):
        graphs = get_iterator(graph, self.num_groups)
        for i, graph in enumerate(graphs):
            getattr(self.components, str(i)).reset_graph(graph)
        self.graph = graphs
                
    def reset_out_indices(self, out_indices=None):
        num_groups = len(self.group_index)
        out_indices = get_iterator(out_indices, num_groups)
        for i in range(num_groups):
            getattr(self.components, str(i)).reset_out_indices(out_indices[i])
        self.out_indices = out_indices
                
    def forward(self, x):
        if self.group_index is None or len(self.group_index)==1:
            return self.components[0](x)
        N = x.size(0) if self.out_indices is None else len(self.out_indices)
        out = Variable(torch.zeros(N, functools.reduce(lambda x,y:x+y, self.out_dim)).type(dtype['float']))
            
        j = 0
        for i, idx in enumerate(self.group_index):
            out[:, j:j+self.out_dim[i]] = self.components[i](x[:,idx])
            j += self.out_dim[i]
            
        if self.merge:
            out_dim = self.merge_dim
            num_groups = len(self.out_dim)
            y = Variable(torch.zeros(N, out_dim).type(dtype['float']))
            if self.merge_type == 'sum':
                # normalize group weight
                self.group_weight_normalized = nn.functional.softmax(self.group_weight, dim=0)
                # Warning: cannot change y inplace, eg. y += something (and y = y+something?)
                y = (self.group_weight_normalized.unsqueeze(1) * out.view(N, num_groups, out_dim)).sum(1)
            elif self.merge_type == 'affine':
                y = nn.functional.linear(out, self.feature_weight)
            return y
        else:
            return out
        

class MultiviewAttention(nn.Module): # This is AffinityNet model used in the Bioinformatics paper
    r"""Stack GraphAttentionGroup layers; 
        For simplicity, assume for each layer, the parameters of each group has the same shape
    
    Args:
        Has the same interface with GraphAttentionGroup, except
            merge: a list of bool variable; default None, set it [False, False, ..., False, True] internally 
            hidden_dims: must be an iterable of int (len(hidden_dims) == num_layers) 
                                                or iterable (len(hidden_dims[0]) == num_views)

        Warnings:
            Be careful to use out_indices, feature_subset, can be buggy
           
    Shape:
        - Input: (N, *)
        - Output: 
    
    Attributes:
        Variables of each GraphAttentionGroupLayer
    
    Examples:
    
        >>> m = MultiviewAttention(4, [3,2], group_index=[range(2), range(2,4)])
        >>> x = Variable(torch.randn(1, 4))
        >>> print(m(x))
        >>> model = FeatureExtractor(m.layers, [0,1])
        >>> print(model(x))
    """
    def __init__(self, in_dim, hidden_dims, k=None, graph=None, out_indices=None, 
                 feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                 nonlinearity_2=None, use_previous_graph=True, group_index=None, merge=None,
                merge_type='sum', reset_graph_every_forward=False, no_feature_transformation=False,
                rescale=True, merge_dim=None, layer_norm=False, layer_magnitude=100, 
                key_dim=None):
        super(MultiviewAttention, self).__init__()
        assert isinstance(in_dim, int)
        assert isinstance(hidden_dims, collections.Iterable)
        self.reset_graph_every_forward = reset_graph_every_forward
        self.hidden_dims = hidden_dims
        num_layers = len(hidden_dims)
        self.num_layers = num_layers
        if group_index is None:
            group_index = [range(in_dim)] if feature_subset is None else [feature_subset]
        if merge is None:
            merge = [False]*(num_layers-1) + [True]
        elif isinstance(merge, bool):
            merge = get_iterator(merge, num_layers)
        out_indices = get_iterator(out_indices, num_layers)
        k = get_iterator(k, num_layers)
        no_feature_transformation = get_iterator(no_feature_transformation, num_layers)
        rescale = get_iterator(rescale, num_layers)
        # buggy here: interact with merge
        merge_dim = get_iterator(merge_dim, num_layers)

        if layer_norm is True:
            layer_norm = [True]*(num_layers-1) + [False]
        layer_norm = get_iterator(layer_norm, num_layers)
        layer_magnitude = get_iterator(layer_magnitude, num_layers)
        key_dim = get_iterator(key_dim, num_layers)

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(str(i),
                GraphAttentionGroup(in_dim, hidden_dims[i], k[i], graph, out_indices[i], None, 
                                    kernel, nonlinearity_1, nonlinearity_2, use_previous_graph, 
                                    group_index, merge[i], merge_type, reset_graph_every_forward=False,
                                    no_feature_transformation=no_feature_transformation[i], 
                                    rescale=rescale[i], merge_dim=merge_dim[i],
                                    layer_norm=layer_norm[i], layer_magnitude=layer_magnitude[i],
                                    key_dim=key_dim[i]))
            # Very Very buggy here
            # assume hidden_dims[i] is int or [int, int] 
            h = get_iterator(hidden_dims[i], len(group_index))
            if merge[i]:
                in_dim = h[0] if merge_dim[i] is None else merge_dim[i]
                group_index = [range(in_dim)]
            else:
                in_dim = sum(h)
                group_index = []
                cnt = 0
                for tmp in h:
                    group_index.append(range(cnt,cnt+tmp))
                    cnt += tmp
                    
    def reset_graph(self, graph=None):
        for i in range(self.num_layers):
            getattr(self.layers, str(i)).reset_graph(graph)
        self.graph = graph
                    
    def reset_out_indices(self, out_indices=None):
        num_layers = len(self.hidden_dims)
        out_indices = get_iterator(out_indices, num_layers)
        for i in range(num_layers):
            getattr(self.layers, str(i)).reset_out_indices(out_indices[i])
        self.out_indices = out_indices
                    
    def forward(self, x):
        if self.reset_graph_every_forward:
            self.reset_graph()
            
        return self.layers(x)