import torch.nn as nn
import torch.nn.functional as F


class Factor1d(nn.Module):
  """Similar to masked attention
  
  """
  def __init__(self, in_features, in_dim, out_features, out_dim, adj_mat=None, bias=True):
    super(Factor1d, self).__init__()
    self.linear1 = nn.Linear(in_dim, out_dim, bias) # based on intuition, not justified
    self.linear2 = nn.Linear(out_dim, out_dim, bias)
    self.linear3 = nn.Linear(in_features, out_features, bias)
    self.linear4 = nn.Linear(out_features, out_features, bias)
    self.adj_mat = adj_mat

  def forward(self, x):
    out = F.relu(self.linear2(F.relu(self.linear1(x))).transpose(1, 2)) # (NxDxC -> NxCxD)
    if self.adj_mat is None:
      return self.linear4(F.relu(self.linear3(out))).transpose(1, 2)
    else:
      return self.linear4(F.relu(
        F.linear(out, self.linear3.weight*self.adj_mat.float(), self.linear3.bias))).transpose(1, 2)
