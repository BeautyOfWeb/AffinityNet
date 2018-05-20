import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
  dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor} 
else:
  dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor} 

class MultiheadAttention(nn.Module):
  """
  """
  def __init__(self, in_dim, out_dim, key_dim, value_dim, num_heads=1, mask=False, 
               query_in_dim=None, knn=None):
    super(MultiheadAttention, self).__init__()
    self.key_dim = key_dim
    self.keys = nn.ModuleList([nn.Linear(in_dim, key_dim) for i in range(num_heads)])
    if query_in_dim is not None:
      self.keys_query = nn.ModuleList([nn.Linear(query_in_dim, key_dim) 
                                       for i in range(num_heads)])
    self.values = nn.ModuleList([nn.Linear(in_dim, value_dim) for i in range(num_heads)])
    self.out = nn.Linear(value_dim*num_heads, out_dim)
    self.mask = mask
    self.knn = knn
    
  def forward(self, x, q=None, return_graph=False):
    y = []
    if return_graph:
      graph = []
    if q is not None:
      #assert self.knn is None or self.knn <= x.size(-2) # found bug here, not clear why yet
      size_x = x.size()
      size_q = q.size()
      x = x.contiguous().view(size_x[0], -1, size_x[-1])
      q = q.contiguous().view(size_q[0], -1, size_q[-1])
      self.mask = False
    for i, (K, V) in enumerate(zip(self.keys, self.values)):
      key = K(x)
      value = V(x)
      if q is None:
        query = key
      else:
        if hasattr(self, 'keys_query'):
          query = self.keys_query[i](q)
        else:
          query = K(q)
      att_unnorm = (query.unsqueeze(-2)*key.unsqueeze(-3)).sum(-1) / np.sqrt(self.key_dim)
      if return_graph:
        graph.append(nn.functional.softmax(att_unnorm, dim=-1))
      if self.mask: # mask right side; useful for decoder with sequential output
        seq_len = att_unnorm.size(-2)
        if att_unnorm.dim() == 3:
          for i in range(seq_len-1):
            att_unnorm[:, i, (i+1):] = float('-inf')
        elif att_unnorm.dim() == 4:
          for i in range(seq_len-1):
            att_unnorm[:, :, i, (i+1):] = float('-inf')
        else:
          raise ValueError('Expect x.dim() <= 4, but x.dim() = {0}'.format(x.dim()))
      if isinstance(self.knn, int):
        self.knn = min(self.knn, att_unnorm.size(-1))
        att_topk, idx = att_unnorm.topk(self.knn, dim=-1)
        att_ = Variable(torch.zeros(att_unnorm.size()).fill_(float('-inf')).type(dtype['float'])) 
        att_.scatter_(-1, idx, att_topk)
        att_unnorm = att_
      att = nn.functional.softmax(att_unnorm, dim=-1)
      cur_y = (att.unsqueeze(-1) * value.unsqueeze(-3)).sum(-2)
      if q is not None:
        cur_y = cur_y.contiguous().view(*size_q[:-1], cur_y.size(-1))
      y.append(cur_y)   
    y = torch.cat(y, -1)
    y = self.out(y)
    if return_graph:
      graph = torch.stack(graph).mean(0)
      return y, graph
    return y

  
class EncoderAttention(nn.Module):
  """
  """
  def __init__(self, in_dim, out_dim, key_dim, value_dim, fc_dim, num_heads=1, residual=True,
              normalization=None, nonlinearity=nn.ReLU(), mask=False, query_in_dim=None, knn=None):
    super(EncoderAttention, self).__init__()
    self.attention = MultiheadAttention(in_dim, out_dim, key_dim, value_dim, num_heads, mask=mask, 
                                        query_in_dim=query_in_dim, knn=knn)
    self.residual = residual
    self.normalization = normalization
    self.fc = nn.Sequential(nn.Linear(out_dim, fc_dim),
                           nonlinearity,
                           nn.Linear(fc_dim, out_dim))

  def forward(self, x, q=None, return_graph=False):
    if return_graph:
      out, graph = self.attention(x, q, return_graph=True)
    else:
      out = self.attention(x, q)
    if self.residual:
      out += x
    if isinstance(self.normalization, nn.Module):
      out = self.normalization(out)
    x = self.fc(out)
    if self.residual:
      x += out
    if isinstance(self.normalization, nn.Module):
      out = self.normalization(x)
    if return_graph:
      return out, graph
    return out
  
class DecoderAttention(nn.Module):
  """
  """
  def __init__(self, in_dim, out_dim, key_dim, value_dim, fc_dim, num_heads=1, residual=True,
              normalization=None, nonlinearity=nn.ReLU(), mask=True, query_key=False, knn=None):
    super(DecoderAttention, self).__init__()
    if residual:
      assert in_dim == out_dim
    self.attention_mask = MultiheadAttention(in_dim, out_dim, key_dim, value_dim, num_heads, mask=mask, knn=knn)
    self.attention_encoder = MultiheadAttention(in_dim, out_dim, key_dim, value_dim, num_heads, 
                                                mask=False, 
                                                query_in_dim=out_dim if query_key else None, knn=knn)
    self.residual = residual
    self.normalization = normalization
    self.fc = nn.Sequential(nn.Linear(out_dim, fc_dim),
                           nonlinearity,
                           nn.Linear(fc_dim, out_dim))
    
  def forward(self, x, input, return_graph=False):
    if return_graph:
      out, graph = self.attention_mask(x, return_graph=True)
    else:
      out = self.attention_mask(x)
    if self.residual:
      out = out + x
    if isinstance(self.normalization, nn.Module):
      out = self.normalization(out)
    x = self.attention_encoder(input, out)
    if self.residual:
      out = out + x
    if isinstance(self.normalization, nn.Module):
      out = self.normalization(out)
    x = self.fc(out)
    if self.residual:
      x = x + out
    if isinstance(self.normalization, nn.Module):
      out = self.normalization(x)
    if return_graph:
      return out, graph
    return out
  

def get_uniq_topk(rank, history):
  res = []
  if history is None:
    res = rank[:, 0]
    history = rank[:, :1]
  else:
    for r, h in zip(rank.data, history.data):
      for i in r:
        if i in h:
          continue
        else:
          res.append(i)
          break
    res =  Variable(dtype['long'](res)) 
    history = torch.cat([history, res.unsqueeze(-1)], -1) 
  return res, history

def get_target(s, t):
  return Variable(dtype['long'](np.array([[
    k if k in set(j).intersection(i) else np.random.choice(list(set(j).difference(i))) 
    for idx, k in enumerate(i)] for i,j in zip(s.data, t.data)])))


class Transformer(nn.Module):
  """
  """
  def __init__(self, in_dim, key_dim, value_dim, fc_dim, linear_dim, in_voc_size,
               out_voc_size, in_seq_len, out_seq_len, encode_input_position=True, 
               encode_output_position=False, num_heads=1, num_attention=1, residual=True, 
               normalization=None, nonlinearity=nn.ReLU(), duplicated_attention=False, mask=True,
              unique_output=False, knn=None):
    super(Transformer, self).__init__()
    self.in_dim = in_dim
    self.out_seq_len = out_seq_len
    self.out_voc_size = out_voc_size
    self.in_embed = nn.Embedding(in_voc_size, in_dim)
    self.out_embed = nn.Embedding(out_voc_size+2, in_dim)
    self.encode_input_position = encode_input_position
    if self.encode_input_position:
      self.input_pos_weight = nn.Parameter(torch.randn(2)) 
      self.input_pos_vec = Variable(torch.Tensor([[np.sin(i/in_seq_len**(j/in_dim)) if j%2==0 
                               else np.cos(i/in_seq_len**(j/in_dim)) 
                     for j in range(in_dim)] for i in range(in_seq_len)]).type(dtype['float']))   
    self.encode_output_position = encode_output_position
    if self.encode_output_position:
      self.output_pos_weight = nn.Parameter(torch.randn(2)) 
      self.output_pos_vec = Variable(torch.Tensor([[np.sin(i/out_seq_len**(j/in_dim)) if j%2==0 
                               else np.cos(i/out_seq_len**(j/in_dim)) 
                     for j in range(in_dim)] for i in range(out_seq_len)]).type(dtype['float'])) 
    
    if duplicated_attention:
      self.encoders = nn.ModuleList([EncoderAttention(
        in_dim, in_dim, key_dim, value_dim, fc_dim, num_heads, residual, normalization, nonlinearity, knn=knn)] 
                                    * num_attention)    
      self.decoders = nn.ModuleList([DecoderAttention(
        in_dim, in_dim, key_dim, value_dim, fc_dim, num_heads, residual, normalization, 
        nonlinearity, mask, knn=knn)] * num_attention)
    else:
      self.encoders = nn.ModuleList()
      self.decoders = nn.ModuleList()
      for i in range(num_attention):
        self.encoders.append(EncoderAttention(
          in_dim, in_dim, key_dim, value_dim, fc_dim, num_heads, residual, normalization, nonlinearity, knn=knn))
        self.decoders.append(DecoderAttention(
          in_dim, in_dim, key_dim, value_dim, fc_dim, num_heads, residual, normalization, 
          nonlinearity, mask, knn=knn))
    
    self.linear = nn.Linear(in_dim, out_voc_size+1)
    self.unique_output = unique_output
    self.knn = knn
    
  def forward(self, x, out=None, sequential=True, last_output_only=True):
    if sequential:
      assert self.knn is None
    if x.dim()==2:
      x = self.in_embed(x)
    else:
      size_x = x.size()
      x = self.in_embed(x.contiguous().view(-1, size_x[-1])).contiguous().view(*size_x, self.in_dim)
    if self.encode_input_position:
      pos_weight = nn.functional.softmax(self.input_pos_weight, dim=0)
      x = x*pos_weight[0] + self.input_pos_vec*pos_weight[1]  
    for encoder in self.encoders:
      x = encoder(x)
      
    if not sequential:
      # This does not work well
      if out is None:
        out = Variable(dtype['long']([[self.out_voc_size+1]*self.out_seq_len]*x.size(0))) 
      out = self.out_embed(out)
      if self.encode_output_position:
        pos_weight = nn.functional.softmax(self.output_pos_weight, dim=0)
        out = out*pos_weight[0] + self.output_pos_vec*pos_weight[1]
      for decoder in self.decoders:
        cur_out = decoder(out, x)
      y = self.linear(cur_out)

    else:
      cur_out = self.out_embed(Variable(dtype['long']([self.out_voc_size]*x.size(0)).
                                        unsqueeze(-1))) 
      y = []
      if self.unique_output:
        self.seq_generated = None
      for i in range(self.out_seq_len):
        for decoder in self.decoders:
          cur_out = decoder(cur_out, x)
        cur_y = self.linear(cur_out)[:, -1]
        y.append(cur_y)
        if self.unique_output:
          assert self.out_seq_len <= self.out_voc_size+1
          rank = cur_y.topk(self.out_seq_len, dim=-1)[1]
          idx, self.seq_generated = get_uniq_topk(rank, self.seq_generated)
          next_out = self.out_embed.weight[idx]
        else:
          next_out = self.out_embed.weight[cur_y.topk(1, dim=-1)[1].squeeze()]
        if self.encode_output_position:
          pos_weight = nn.functional.softmax(self.output_pos_weight, dim=0)
          next_out = next_out*pos_weight[0] + self.output_pos_vec[i]*pos_weight[1]
        cur_out = torch.cat([cur_out, next_out.unsqueeze(-2)], dim=-2)
      y = torch.stack(y, dim=-2)
    return y


class StackedEncoder(nn.Module):
  """

  Examples:
  
  model = StackedEncoder(in_dim=4, key_dim=3, value_dim=5, fc_dim=6, linear_dim=7, num_cls=8, num_heads=2, num_attention=2, 
      knn=None, residual=True, normalization=None, nonlinearity=nn.ReLU(), duplicated_attention=False, mask=False)
  x = Variable(torch.randn(4, 4))
  model(x, return_graph=True, return_all=True)
  """
  def __init__(self, in_dim, key_dim, value_dim, fc_dim, linear_dim, num_cls, num_heads=1, num_attention=1, 
    knn=None, residual=True, normalization=None, nonlinearity=nn.ReLU(), duplicated_attention=False, mask=False,
    return_graph=False, return_all=False):
    super(StackedEncoder, self).__init__()

    if duplicated_attention:
      self.encoders = nn.ModuleList([EncoderAttention(
        in_dim, in_dim, key_dim, value_dim, fc_dim, num_heads, residual, normalization, nonlinearity, knn=knn)] 
                                    * num_attention)
    else:
      self.encoders = nn.ModuleList()
      for i in range(num_attention):
        self.encoders.append(EncoderAttention(
          in_dim, in_dim, key_dim, value_dim, fc_dim, num_heads, residual, normalization, nonlinearity, knn=knn))
    self.linear = nn.Linear(in_dim, num_cls)
    self.return_graph = return_graph
    self.return_all = return_all

  def forward(self, x):
    return_graph = self.return_graph
    return_all = self.return_all
    if return_graph and return_all:
      graphs = []
    for encoder in self.encoders:
      if return_graph:
        x, graph = encoder(x, return_graph=True)
        if return_all:
          graphs.append(graph)
      else:
        x = encoder(x)
    out = self.linear(x)
    if return_graph:
      if return_all:
          return out, graphs
      else:
        return out, graph
    else:
      return out