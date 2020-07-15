import torch.nn.functional as F
import torch

logits = torch.randn(32, 1)
logits_right = torch.tensor(1.0) - logits
comb = torch.cat((logits, logits_right), dim = -1)

 # Sample soft categorical using reparametrization trick:
a = F.gumbel_softmax(comb, tau=0.1)

print(comb)
print(a)

right = torch.ones(3,2).unsqueeze(0).unsqueeze(-1)
left = torch.tensor(1.0) - right

# print(right)
# # print(left)
#
# comb = torch.cat((right, left), dim = -1)
# tiny = comb[0,:,:, 0]
# print(comb.shape)
# print(comb)
# print(tiny.shape)
# print(tiny)

