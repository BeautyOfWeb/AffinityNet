import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.metrics
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import spectral_clustering, KMeans

import torch
import torch.nn as nn
from torch.autograd import Variable

from .graph_attention import *

if torch.cuda.is_available():
  dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor} 
else:
  dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor} 


def pca(x, n_components=2, verbose=False):
    r"""PCA for 2-D visualization
    """
    if len(x)>10000:
        pca = IncrementalPCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    if isinstance(x, Variable):
        x = x.cpu().numpy().copy()
    pca.fit(x)
    if verbose:
        print(pca.explained_variance_, pca.noise_variance_)
        plt.title('explained_variance')
        plt.plot(pca.explained_variance_.tolist() + [pca.noise_variance_], 'ro')
        plt.show()
    return pca.fit_transform(x)

def plot_scatter(y_=None, model_=None, x_=None, title='', labels=None, colors=None, size=15, 
                 marker_size=20, folder='.', save_fig=False):
    r"""2D scatter plot
    """
    if y_ is None:
        assert model_ is not None and x_ is not None
        y_ = model_(x_.contiguous())
    if colors is not None:
        assert len(colors) == len(y_)
    else:
        if labels is not None:
            assert len(y_) == len(labels)
            color = sorted(matplotlib.colors.BASE_COLORS)
            colors = [color[i] for i in labels]
    if isinstance(y_, Variable):
        y_ = y_.data.cpu().numpy()
    if y_.shape[1] > 2:
        y_ = pca(y_)
    plt.figure(figsize=(size, size))
    plt.scatter(y_[:,0],y_[:,1], c=colors, s=marker_size)
    if save_fig:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder+'/'+title+'.png', bbox_inches='tight', dpi=200)
    else:
        plt.title(title)
        plt.show()
    plt.close()

    
def cal_nmi(y_true, y_pred=None, mat=None, num_clusters=2, return_value=True, verbose=False):
    r"""Calculate accuracy, NMI, and confusion matrix
    """
    if y_pred is None:
        assert mat is not None
        if isinstance(mat, Variable):
            mat = mat.cpu().numpy()
        y_pred = spectral_clustering(affinity=mat, n_clusters=num_clusters)
    if isinstance(y_true, Variable):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Variable):
        y_pred = y_pred.cpu().numpy()
    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    nmi = sklearn.metrics.adjusted_mutual_info_score(labels_true=y_true, labels_pred=y_pred)
    confusion_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
    if verbose:
        print('acc={0}, nmi={1}, \n{2}'.format(acc, nmi, confusion_mat))
    if return_value:
        return acc, nmi, confusion_mat
    
def eval_acc(model, x_var, labels, y_pred=None, return_value=True, verbose=False):
    r"""Calculate accuracy, NMI and confusion matrix
    """
    if isinstance(labels, Variable):
        labels = labels.cpu().numpy().copy()
    if y_pred is None:
        y_pred = model(x_var)
    labels_pred = y_pred.topk(k=1)[1].cpu().numpy().reshape(-1)
    acc = sklearn.metrics.accuracy_score(y_true=labels, y_pred=labels_pred)
    nmi = sklearn.metrics.adjusted_mutual_info_score(labels_true=labels, labels_pred=labels_pred)
    confusion_mat = sklearn.metrics.confusion_matrix(labels, labels_pred)
    if verbose:
        print('acc={0}, nmi={1}, \n{2}'.format(acc, nmi, confusion_mat))
    if return_value:
        return acc, nmi, confusion_mat

def visualize_val(X_val, y_val, solver, batch_size=None, title='X_val', topk=1, save_fig=False, save_folder='',
                 figsize=10, return_value=True, silent=True):
    r"""2D scatter plot before and after training
    """
    if not silent:
        if batch_size is None:
            batch_size = X_val.size(0)
        title_ = 'before training {0}'.format(title)
        plot_scatter(X_val, title=title_, colors=y_val, folder=save_folder, save_fig=save_fig, 
                    size=figsize)
    
    y = solver.predict(batch_size=batch_size, save_file=False, 
                       X=X_val, y=y_val, topk=topk)
    acc, nmi, confusion_mat = eval_acc(None, None, y_val, y_pred=y)
    labels_pred = y.topk(k=1)[1].cpu().numpy().reshape(-1)
    if y_val.max()>0 and labels_pred.max()>0:
        num_cls = y_val.max() + 1
        f1_score = sklearn.metrics.f1_score(y_true=y_val.cpu().numpy(), y_pred=labels_pred, 
            average='binary' if num_cls==2 else 'weighted')
    else:
        f1_score = 0

    if not silent:
        title_ = 'after training {0} acc={1}'.format(title, acc)
        plot_scatter(y, title=title_, colors=y_val, folder=save_folder, save_fig=save_fig, 
                    size=figsize)
    if return_value:
        return acc, nmi, confusion_mat, f1_score
    
def test_regression(x, y, model, model_true=None, num_iters=50, lr=1, lr_decay=0.2, lr_decay_every=10, 
                    loss_fn=nn.MSELoss(), verbose=True, print_param=True, retain_graph=True,
                   loss_title='loss', folder='.', save_fig=False, size=15):
    r"""Train the model
    """
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    losses = []
    for i in range(num_iters):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        losses.append(loss.item())
        if (i+1) % lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            if verbose:
                print(i, loss.item())
                if model_true is not None:
                    for name, param in model_true.named_parameters():
                        # getattr(model, attr) works only when attr does not contain '.'
                        param_ = functools.reduce(lambda model, a: getattr(model,a), name.split('.'), model)
                        print('{0} dist={1}'.format(
                            name, torch.dist(param.data, param_.data)))
                        if print_param:
                            print('true={0}, learned={1}'.format(
                                param.cpu().numpy(), param_.cpu().numpy()))
        if i==num_iters-1 and isinstance(y.data, dtype['long']):
            eval_acc(model, x, y, y_pred=y_pred)
    plt.figure(figsize=(size, size))
    plt.plot(losses, 'm:')
    plt.xlabel('number of iterations')
    plt.ylabel('loss')
    if save_fig:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder+'/'+loss_title+'.png', bbox_inches='tight', dpi=200)
    else:
        plt.title(loss_title)
        plt.show()
    plt.close()

    
def randperm(idx, random_examples=False, seed=None):
    """Randomly permute indices
    
    Args:
        idx: torch.LongTensor, indices to be permuted
        random_examples: bool, if True, return a random permutation
        seed: if int, then set seed before random permutation
    
    """
    n = len(idx)
    if isinstance(seed, int):
        torch.manual_seed(seed)
        return idx[torch.randperm(n)]
    if random_examples:
        return idx[torch.randperm(n)]
    else:
        return idx

    
def split_train_test(x_var, y_var, train_indices, y_true=None, seed=None):
    r"""Split data into training and test (validation) set
    
    Arg:
        x_var, y_var: Variable or torch.Tensor, the first dimension will be splitted
        train_indices: torch.LongTensor
        y_true: y_test = y_var[test_indices] if y_true is None else y_true[test_indices]
        
    Returns:
    
    Examples:
    
        >>>
    """
    test_indices = dtype['long'](sorted(set(range(x_var.size(0))).difference(train_indices.cpu().numpy())))
    if seed is not None:
        train_indices = randperm(train_indices, random_examples=True, seed=seed)
        test_indices = randperm(test_indices, random_examples=True, seed=seed)
    x_train = x_var[train_indices]
    y_train = y_var[train_indices]
    x_test = x_var[test_indices]
    if y_true is None:
        y_test = y_var[test_indices]
    else:
        y_test = y_true[test_indices]
    return x_train, y_train, x_test, y_test, train_indices, test_indices    


def split_data(x_var, y_var, num_examples=1, proportions=None, seed=None, random_examples=False):
    num_clusters = y_var.max().item() + 1 # assume y_var is LongTensor starting from 0 to num_cls-1
    if proportions is not None:
        if isinstance(proportions, float):
            assert proportions > 0 and proportions < 1
            proportions = [proportions]*num_clusters
        num_examples = [max(1,round(torch.nonzero(y_var==i).size(0) * proportions[i])) for i in range(num_clusters)]
    if isinstance(num_examples, int):
        num_examples_per_class = num_examples
        num_examples = [num_examples_per_class]*num_clusters
    assert num_clusters == len(num_examples)
    train_indices = [randperm(torch.nonzero(y_var==i), random_examples, seed)[:num_examples[i],0]
                     for i in range(num_clusters)]
    train_indices = torch.cat(train_indices, dim=0).data
    return split_train_test(x_var, y_var, train_indices, seed=seed)

def split_train_val_test(x_var, y_var, proportions, seed=None, random_examples=False, 
                        train_val_test=False):
  n = x_var.size(0)
  idx = randperm(dtype['long'](range(n)), seed=seed, random_examples=random_examples)
#   assert sum(proportions)==1, 'proportions should sum to 1!'
  split = [round(n*p) for p in proportions]
  xs = []
  ys = []
  indices = []
  start = 0
  for s in split[:-1]:
    xs.append(x_var[idx[start:start+s]])
    ys.append(y_var[idx[start:start+s]])
    indices.append(idx[start:start+s])
    start += s
  xs.append(x_var[idx[start:]])
  ys.append(y_var[idx[start:]])
  indices.append(idx[start:])
  if train_val_test:
    assert len(proportions)==3
    return xs[0], ys[0], indices[0], xs[1], ys[1], indices[1], xs[2], ys[2], indices[2]
  return xs, ys, indices

def construct_linear_model(in_dim, hidden_dims, num_groups=1, nonlinearity=nn.ReLU()):
    r"""Construct a multi-layer linear model
    
    Args:
        in_dim: input dimension
        hidden_dims: iterable of int, number of hidden units in each layer
        num_groups: int, if > 1, add a Weighted view after input layer
        nonlinearity: nonlinear activations after each Linear layer
        
    Returns:
        model of nn.Module
        
    Examples:
    
        >>> construct_linear_model(10, [10])
    """
    model = nn.Sequential()
    if num_groups > 1:
        model.add_module('weightedview', WeightedView(num_groups))
        in_dim = in_dim // num_groups
    model.add_module('linear0', nn.Linear(in_dim, hidden_dims[0]))
    model.add_module('activation0', nonlinearity)
    for i in range(1, len(hidden_dims)):
        model.add_module('linear'+str(i), nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        model.add_module('activation'+str(i), nonlinearity)
    return model
    
    
def example_learning(x_var, y_var, num_examples=1, num_clusters=2, num_groups=1, hidden_dims=[50],
                     nonlinearity=nn.ReLU(), with_last_nonlinearity=False, num_iters=50, lr=1, lr_decay=0.2,
                     lr_decay_every=10, model=None, model_head=None, return_model=False, pca_dim=None,
                     random_examples=False, seed=None, y_true=None, folder='.', save_fig=False,
                    marker_size=20, return_value=True, x_new=None, y_new=None):
    r"""Few-shot training/learning
    """
    assert isinstance(x_var, Variable) and isinstance(y_var, Variable)
    assert y_true is None or isinstance(y_true, Variable)
        
    num_examples_per_class = num_examples
    if isinstance(num_examples_per_class, int):
        num_examples = [num_examples_per_class]*num_clusters
    examples_indices = [randperm(torch.nonzero(y_var==i), random_examples, seed)[:num_examples[i],0] 
                        for i in range(num_clusters)]
    # In the following line, if '.data' is missing, it will be wrong because out_indices will be Variable
    train_indices = torch.cat(examples_indices).data
    if len(train_indices) <= 10:
        print('Examples (indices) to train', train_indices.cpu().numpy().tolist())

    x_train, y_train, x_test, y_test, train_indices, test_indices = split_train_test(
        x_var, y_var, train_indices, y_true)
    
    num_features_per_view = x_var.size(1)//num_groups
    if isinstance(pca_dim, int):
        assert pca_dim < num_features_per_view and pca_dim > 0
        x_pca = np.concatenate([pca(x_var[:,i*num_features_per_view:(i+1)*num_features_per_view], pca_dim)
                                for i in range(num_groups)], axis=1)
        x_var_pca = Variable(torch.from_numpy(x_pca).type(dtype['float']))
        x_train_pca, _, x_test_pca, _, _, _ = split_train_test(x_var_pca, y_var, train_indices, y_true)
        
    color = sorted(matplotlib.colors.BASE_COLORS)
    color.remove('w')
    color = np.array(color)
    if y_true is None:
        colors = np.array([color[i] for i in y_var.data])
    else:
        colors = np.array([color[i] for i in y_true.data])
    j = -1
    for ex in examples_indices:
        colors[ex.cpu().numpy()] = color[j]
        j = j-1
        
    marker_sizes = np.array([marker_size]*x_var.size(0))
    marker_sizes[train_indices.cpu().numpy()] = int(marker_size*1.5)
        
    in_dim = x_var.size(1)
    if model is None or model_head is None:
        model_head = construct_linear_model(in_dim, hidden_dims, num_groups, nonlinearity)
        model = construct_linear_model(in_dim, hidden_dims, num_groups, nonlinearity)
        model.add_module('linear'+str(len(hidden_dims)), nn.Linear(hidden_dims[-1], num_clusters))
        if with_last_nonlinearity:
            model_head.add_module('activation'+str(len(hidden_dims)), nonlinearity)
            model.add_module('activation'+str(len(hidden_dims)), nonlinearity)
            
    get_partial_model(model_head, model)
    plot_scatter(model_=model_head, x_=x_var, title='Before training (2nd to last layer)', colors=colors, 
                folder=folder, save_fig=save_fig, marker_size=marker_sizes)
    plot_scatter(model_=model, x_=x_var, title='Before training (output layer)', colors=colors,
                folder=folder, save_fig=save_fig, marker_size=marker_sizes)
    
    if isinstance(pca_dim, int):
        in_dim = pca_dim*num_groups
        pca_model_head = construct_linear_model(in_dim, hidden_dims, num_groups, nonlinearity)
        pca_model = construct_linear_model(in_dim, hidden_dims, num_groups, nonlinearity)
        pca_model.add_module('linear'+str(len(hidden_dims)), nn.Linear(hidden_dims[-1], num_clusters))
        if with_last_nonlinearity:
            pca_model_head.add_module('activation'+str(len(hidden_dims)), nonlinearity)
            pca_model.add_module('activation'+str(len(hidden_dims)), nonlinearity)
        get_partial_model(pca_model_head, pca_model)
        
    if num_examples[0] < 5:
        print('Before training: y_var:', model(x_train).cpu().numpy())
    print('Training in total (depending on y_var) {0} examples: {1}'.format(sum(num_examples), num_examples))
    test_regression(x_train, y_train, model, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                    num_iters=num_iters, lr=lr, lr_decay=lr_decay, lr_decay_every=lr_decay_every,
                   loss_title='training', folder=folder, save_fig=save_fig)
    if num_examples[0] < 5:
        print('After training: y_var:', model(x_train).cpu().numpy())
    get_partial_model(model_head, model)
    plot_scatter(model_=model_head, x_=x_var, title='After training (2nd to last layer)', colors=colors,
                folder=folder, save_fig=save_fig, marker_size=marker_sizes)
    plot_scatter(model_=model, x_=x_var, title='After training (output layer)', colors=colors,
                folder=folder, save_fig=save_fig, marker_size=marker_sizes)
    print('Train acc:')
    eval_acc(model, x_train, y_train)
    if y_true is not None:
        print('Real training acc:')
        eval_acc(model, x_train, y_true[train_indices])
    print('Test acc:')
    res_test = eval_acc(model, x_test, y_test)  
        
    print('All acc:')
    if y_true is None:
        res_all = eval_acc(model, x_var, y_var)
    else:
        res_all = eval_acc(model, x_var, y_true)
        
    if y_new is not None:
        res_new = eval_acc(model, x_new, y_new)
    
    if isinstance(pca_dim, int):
        print('train pca model with {0} examples'.format(len(train_indices)))
        test_regression(x_train_pca, y_train, pca_model, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                        num_iters=num_iters, lr=lr, lr_decay=lr_decay, lr_decay_every=lr_decay_every,
                       loss_title='training_pca', folder=folder, save_fig=save_fig)
        get_partial_model(pca_model_head, pca_model)
        plot_scatter(model_=pca_model_head, x_=x_var_pca, title='After training PCA (2nd to last layer)',
                     colors=colors, folder=folder, save_fig=save_fig, marker_size=marker_sizes)
        plot_scatter(model_=pca_model, x_=x_var_pca, title='After training PCA (output layer)', colors=colors,
                    folder=folder, save_fig=save_fig, marker_size=marker_sizes)
        print('Train PCA acc:')
        eval_acc(pca_model, x_train_pca, y_train)
        if y_true is not None:
            print('Real training PCA acc:')
            eval_acc(pca_model, x_train_pca, y_true[train_indices])
        print('Test PCA acc:')
        eval_acc(pca_model, x_test_pca, y_test)
        print('All PCA acc:')
        if y_true is None:
            eval_acc(pca_model, x_var_pca, y_var)
        else:
            eval_acc(pca_model, x_var_pca, y_true)
    
    num_clusters_trained = num_clusters
    if y_true is None:
        num_clusters = len(np.unique(y_var.cpu().numpy()))
    else:
        num_clusters = len(np.unique(y_true.cpu().numpy()))
    if num_clusters > num_clusters_trained:
        if isinstance(num_examples_per_class, int):
            num_examples = [num_examples_per_class] * num_clusters
        # the case when we provide num_examples = [5,5], but num_class = 3
        if len(num_examples) < num_clusters:
            num_examples = num_examples + [min(num_examples)] * (num_clusters-len(num_examples))
        examples_indices = [randperm(torch.nonzero(y_var == i), random_examples, seed)[:num_examples[i],0] 
                            for i in range(num_clusters)]
        train_indices = torch.cat(examples_indices).data
        x_train = x_var[train_indices]
        y_train = y_var[train_indices]
        
        print('Finetune in total (depending on y_var) {0} examples: {1}'.format(sum(num_examples), num_examples))
        print('Finetune 2nd to last layer:')
        model_finetune = FineTuneModel(model_head, nn.Linear(hidden_dims[-1], num_clusters))
        test_regression(x_train, y_train, model_finetune, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                        lr=lr, num_iters=num_iters, lr_decay=lr_decay, lr_decay_every=lr_decay_every,
                       loss_title='finetune_2nd', folder=folder, save_fig=save_fig)
        print('After finetune 2nd to last:')
        if y_true is None:
            res_all_finetune_2nd = eval_acc(model_finetune, x_var, y_var)
        else:
            res_all_finetune_2nd = eval_acc(model_finetune, x_var, y_true)
            
        if y_new is not None:
            res_new_finetune_2nd = eval_acc(model_finetune, x_new, y_new)
        
        print('Finetune the last layer')
        model_finetune = FineTuneModel(model, nn.Linear(num_clusters_trained, num_clusters))
        test_regression(x_train, y_train, model_finetune, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                       lr=lr, num_iters=num_iters, lr_decay=lr_decay, lr_decay_every=lr_decay_every,
                       loss_title='finetune_last', folder=folder, save_fig=save_fig)
        print('After finetune the last layer:')
        if y_true is None:
            res_all_finetune_last = eval_acc(model_finetune, x_var, y_var)
        else:
            res_all_finetune_last = eval_acc(model_finetune, x_var, y_true)
            
        if y_new is not None:
            res_new_finetune_last = eval_acc(model_finetune, x_new, y_new)
        
        if isinstance(pca_dim, int):
            x_train_pca = x_var_pca[train_indices]
            print('Finetune PCA 2nd to last layer:')
            model_finetune = FineTuneModel(pca_model_head, nn.Linear(hidden_dims[-1], num_clusters))
            test_regression(x_train_pca, y_train, model_finetune, print_param=False,
                            loss_fn=nn.CrossEntropyLoss(), lr=lr, num_iters=num_iters, lr_decay=lr_decay,
                            lr_decay_every=lr_decay_every, 
                            loss_title='finetune_2nd_pca', folder=folder, save_fig=save_fig)
            print('After finetune PCA 2nd to last:')
            if y_true is None:
                eval_acc(model_finetune, x_var_pca, y_var)
            else:
                eval_acc(model_finetune, x_var_pca, y_true)

            print('Finetune PCA the last layer')
            model_finetune = FineTuneModel(pca_model, nn.Linear(num_clusters_trained, num_clusters))
            test_regression(x_train_pca, y_train, model_finetune, print_param=False,
                            loss_fn=nn.CrossEntropyLoss(), lr=lr, num_iters=num_iters, lr_decay=lr_decay,
                            lr_decay_every=lr_decay_every,
                           loss_title='finetune_last_pca', folder=folder, save_fig=save_fig)
            print('After finetune PCA the last layer:')
            if y_true is None:
                eval_acc(model_finetune, x_var_pca, y_var)
            else:
                eval_acc(model_finetune, x_var_pca, y_true)
            
    print('spectral clustering using the 2nd to last layer:')
    new_features = model_head(x_var)
    w = torch.norm(new_features-new_features[:,None], dim=-1).exp().cpu().numpy()
    w = knn_graph(w, k=10)
    if y_true is None:
        cal_nmi(y_var, mat=w, num_clusters=num_clusters)
    else:
        cal_nmi(y_true, mat=w, num_clusters=num_clusters)
    print('spectral clustering using the last layer:')
    new_features = model(x_var)
    w = torch.norm(new_features-new_features[:,None], dim=-1).exp().cpu().numpy()
    w = knn_graph(w, k=10)
    if y_true is None:
        cal_nmi(y_var, mat=w, num_clusters=num_clusters)
    else:
        cal_nmi(y_true, mat=w, num_clusters=num_clusters)
    
    if num_groups > 1:
        print('normalized view weight', getattr(model, 'weightedview').normalized_weight)
        mat = getattr(model, 'weightedview')(x_var)
        plot_scatter(mat, colors=colors, title='learned weighted mat', folder=folder, save_fig=save_fig,
                     marker_size=marker_sizes)
        if y_true is None:
            cal_nmi(y_true=y_var, mat=mat, num_clusters=num_clusters)
        else:
            cal_nmi(y_true=y_true, mat=mat, num_clusters=num_clusters)
        num_features_per_view = x_var.size(1) // num_groups
        j = 0
        for i in range(num_groups):
            mat = x_var[:, j:j+num_features_per_view]
            plot_scatter(mat, colors=colors, title='view'+str(i), folder=folder, save_fig=save_fig,
                         marker_size=marker_sizes)
            if y_true is None:
                cal_nmi(y_true=y_var, mat=mat, num_clusters=num_clusters)
            else:
                cal_nmi(y_true=y_true, mat=mat, num_clusters=num_clusters)
            j += num_features_per_view
        mat = x_var.view(x_var.size(0), num_groups, num_features_per_view).mean(1)
        plot_scatter(mat, colors=colors, title='combine view with uniform weight', 
                     folder=folder, save_fig=save_fig, marker_size=marker_sizes)
        if y_true is None:
            cal_nmi(y_true=y_var, mat=mat, num_clusters=num_clusters)
        else:
            cal_nmi(y_true=y_true, mat=mat, num_clusters=num_clusters)
    plot_scatter(x_var, colors=colors, title='x_var(all views concatenated)', 
                folder=folder, save_fig=save_fig, marker_size=marker_sizes)
    if return_model:
        return model, model_head
    
    if return_value:
        if y_new is None:
            if num_clusters > num_clusters_trained:
                return res_test, res_all, res_all_finetune_2nd, res_all_finetune_last
            else:
                return res_test, res_all
        else:
            if num_clusters > num_clusters_trained:
                return (res_test, res_all, res_all_finetune_2nd, res_all_finetune_last,
                        res_new, res_new_finetune_2nd, res_new_finetune_last)
            else:
                return res_test, res_all, res_new
    
    
def clustering(x_var, y_var, num_examples=1, num_clusters=2, hidden_dims=[10,5,2],
               Model=GraphAttentionModel, num_iters=50, lr=1, lr_decay=0.2, lr_decay_every=10):
    assert isinstance(x_var, Variable) and isinstance(y_var, Variable)
    examples_indices = [torch.nonzero(y_var == i)[:num_examples,0] for i in range(num_clusters)]
    # In the following line, if '.data' is missing, it will be wrong because out_indices will be Variable
    out_indices = torch.cat(examples_indices).data
    y_truth = y_var[out_indices]
    color = sorted(matplotlib.colors.BASE_COLORS)
    color.remove('w')
    color = np.array(color)
    colors = np.array([color[i] for i in y_var.data])
    j = -1
    for ex in examples_indices:
        colors[ex.cpu().numpy()] = color[j]
        j = j-1
        
    plot_scatter(x_var, colors=colors, title='x_var')
    
    in_dim = x_var.size(1)
    model = Model(in_dim, hidden_dims, nonlinearities_1=nn.Hardtanh(), nonlinearities_2=None, ks=20,
                  use_previous_graphs=True, out_indices=None)
    if len(hidden_dims) > 1:
        model_head = Model(in_dim, np.array(hidden_dims, dtype=np.int)[:-1].tolist(), 
                           nonlinearities_1=nn.Hardtanh(), nonlinearities_2=None, ks=20, 
                           use_previous_graphs=True, out_indices=None)
        get_partial_model(model_head, model)
    
    if len(hidden_dims) > 1:
        plot_scatter(model_=model_head, x_=x_var, title='Before training (2nd to last layer)', colors=colors)
    plot_scatter(model_=model, x_=x_var, title='Before training (output layer)', colors=colors)
    
    out_indices = [None]*(len(hidden_dims)-1) + [out_indices]
    model.reset_out_indices(out_indices)
    print('Before training: y_var:', model(x_var).cpu().numpy())
    test_regression(x_var, y_truth, model, print_param=False, loss_fn=nn.CrossEntropyLoss(), num_iters=num_iters,
                    lr=lr, lr_decay=lr_decay, lr_decay_every=lr_decay_every)
    print('After training: y_var:', model(x_var).cpu().numpy())
    
    model.reset_out_indices()
    if len(hidden_dims) > 1:
        get_partial_model(model_head, model)
        plot_scatter(model_=model_head, x_=x_var, title='After training (2nd to last layer)', colors=colors)
    plot_scatter(model_=model, x_=x_var, title='After training (output layer)', colors=colors)
    eval_acc(model, x_var, y_var)
    
    num_clusters = len(np.unique(y_var.cpu().numpy()))
    if num_clusters > 2:
        examples_indices = [torch.nonzero(y_var == i)[:num_examples,0] for i in range(num_clusters)]
        out_indices = torch.cat(examples_indices).data
        x_train = x_var[out_indices]
        y_train = y_var[out_indices]
        if len(hidden_dims) > 1:
            model_finetune = FineTuneModel(model_head, nn.Linear(hidden_dims[-2], num_clusters))
            test_regression(x_train, y_train, model_finetune, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                           lr=0.01, num_iters=1)
            print('After finetune 2nd to last:')
            eval_acc(model_finetune, x_var, y_var)

        model_finetune = FineTuneModel(model, nn.Linear(hidden_dims[-1], num_clusters))
        test_regression(x_train, y_train, model_finetune, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                       lr=0.01, num_iters=1)
        print('After finetune last:')
        eval_acc(model_finetune, x_var, y_var)
    
    
    
def test_WeightedFeature(N=20, num_features=10):
    weight = Variable(torch.randn(num_features).type(dtype['float']))
    normalized_weight = torch.nn.functional.softmax(weight, dim=0)
    x = Variable(torch.randn(N, num_features).type(dtype['float']))
    y = x*normalized_weight
    model = WeightedFeature(num_features)
    test_regression(x,y.detach(),model)

    
def test_GraphAttentionLayer(N=20, in_dim=2, out_dim=2, k=None, graph=None, out_indices=None,
                             feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                             nonlinearity_2=None, use_previous_graph=True, 
                             loss_fn=nn.L1Loss(False), print_param=True):
    if isinstance(out_dim, int):
        MODEL = GraphAttentionLayer
    else:
        MODEL = GraphAttentionModel
    model_true = MODEL(in_dim, out_dim, k, graph, out_indices, feature_subset, kernel,
                                     nonlinearity_1, nonlinearity_2, use_previous_graph)
    x = Variable(torch.randn(N, in_dim).type(dtype['float']))
    y = model_true(x)
    model = MODEL(in_dim, out_dim, k, graph, out_indices, feature_subset, kernel,
                                nonlinearity_1, nonlinearity_2, use_previous_graph)
    test_regression(x,y.detach(),model,model_true,loss_fn=loss_fn, print_param=print_param)
    
    
def test_GraphAttentionGroup(N=20, in_dim=4, out_dim=2, k=None, graph=None, out_indices=None,
                             feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                             nonlinearity_2=None, use_previous_graph=True, 
                             group_index=[range(2), range(2,4)], merge=False,
                             loss_fn=nn.L1Loss(False), print_param=True, num_iters=50, lr=0.1, 
                             lr_decay=0.2, lr_decay_every=10, retain_graph=True):
    MODEL = GraphAttentionGroup
    model_true = MODEL(in_dim, out_dim, k, graph, out_indices, feature_subset, kernel,
                       nonlinearity_1, nonlinearity_2, use_previous_graph, group_index, merge)
    x = Variable(torch.randn(N, in_dim).type(dtype['float']))
    y = model_true(x)
    model = MODEL(in_dim, out_dim, k, graph, out_indices, feature_subset, kernel,
                  nonlinearity_1, nonlinearity_2, use_previous_graph, group_index, merge)
    test_regression(x,y.detach(),model,model_true,loss_fn=loss_fn, print_param=print_param, 
                   lr=lr, lr_decay=lr_decay, lr_decay_every=lr_decay_every, retain_graph=retain_graph)
    
    
def test_clustering(N=50, mu=[[0,0], [5,5]], sigma=[2,2], hidden_dims = [3,3,2], 
                    Model=GraphAttentionModel):
    num_clusters = len(mu)
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * num_clusters
    x = []
    labels = []
    for i, (u, s) in enumerate(zip(mu, sigma)):
        x.append(np.random.multivariate_normal(u, np.diag([s,s]), N))
        labels.append([i]*N)
    x = np.concatenate(x, axis=0)
    labels = np.concatenate(labels)       
    color_idx = labels.copy().astype(np.int)
    colors = np.array(sorted(matplotlib.colors.BASE_COLORS))
    colors = colors[color_idx]
    colors[0] = 'y'
    colors[N] = 'r'

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0],x[:,1], c=colors)
    plt.show()
    
    x_var = Variable(torch.from_numpy(x).float().type(dtype['float']))
    out_indices = dtype['long']([0, N])
    out_indices = [None]*(len(hidden_dims)-1) + [out_indices]
    in_dim = 2
    model = Model(in_dim, hidden_dims, nonlinearities_1=nn.Hardtanh(), nonlinearities_2=None, ks=20,
                  use_previous_graphs=True, out_indices=None)
    if len(hidden_dims) > 1:
        model_head = Model(in_dim, np.array(hidden_dims, dtype=np.int)[:-1].tolist(), 
                           nonlinearities_1=nn.Hardtanh(), nonlinearities_2=None, ks=20, 
                           use_previous_graphs=True, out_indices=None)
        get_partial_model(model_head, model)
    
    def plot(model_, x_=x_var, title='', colors=colors, size=5):
        y_test = model_(x_)
        y = y_test.cpu().numpy()
        plt.figure(figsize=(size, size))
        plt.title(title)
        plt.scatter(y[:,0],y[:,1], c=colors)
        plt.show()
    if len(hidden_dims) > 1:
        plot(model_head, x_var, 'Before training (2nd to last layer)')
    plot(model, x_var, 'Before training (output layer)')
    model.reset_out_indices(out_indices)
    y_truth = Variable(dtype['long']([0,1]))
    print('Before training: y_var:', model(x_var).cpu().numpy())
    
    test_regression(x_var, y_truth, model, print_param=False, loss_fn=nn.CrossEntropyLoss())
    y_var = model(x_var)
    print('After training: y_var:', y_var.cpu().numpy())
    model.reset_out_indices()
    if len(hidden_dims) > 1:
        get_partial_model(model_head, model)
        plot(model_head, x_var, 'After training: (2nd to last layer)')
    plot(model, x_var, 'After training: (output layer)')
    def eval_acc(model, x_var, labels):
        y_test = model(x_var)
        labels_pred = y_test.topk(k=1)[1].cpu().numpy().reshape(-1)
        print('acc={0}, nmi={1}, \n{2}'.format(
            sklearn.metrics.accuracy_score(y_true=labels, y_pred=labels_pred),
            sklearn.metrics.adjusted_mutual_info_score(labels_true=labels, labels_pred=labels_pred),
            sklearn.metrics.confusion_matrix(labels, labels_pred)))
    eval_acc(model, x_var, labels)
    
    assert num_clusters > 2
    x_train = x_var[0:N*num_clusters:N]
    y_train = Variable(dtype['long'](range(num_clusters)))
    if len(hidden_dims) > 1:
        model_finetune = FineTuneModel(model_head, nn.Linear(hidden_dims[-2], num_clusters))
        test_regression(x_train, y_train, model_finetune, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                       lr=0.01, num_iters=20)
        print('After finetune 2nd to last:')
        eval_acc(model_finetune, x_var, labels)
    
    model_finetune = FineTuneModel(model, nn.Linear(hidden_dims[-1], num_clusters))
    test_regression(x_train, y_train, model_finetune, print_param=False, loss_fn=nn.CrossEntropyLoss(),
                   lr=0.01, num_iters=20)
    print('After finetune last:')
    eval_acc(model_finetune, x_var, labels)
    
    
def test_MultiviewAttention():
    pass
        