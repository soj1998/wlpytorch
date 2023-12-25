import numpy as np
import torch


inputs = torch.FloatTensor([2])
weights = torch.rand(1, requires_grad=True)
bias = torch.rand(1, requires_grad=True)
t = inputs @ weights
out = t+bias
out.backward()
a = weights.grad
b = bias.grad
c = 1