import torch
import torch.nn as nn

class MyLossFunc(nn.Module):
    def __init__(self):
        super(MyLossFunc, self).__init__()
        return
    def forward(self, prob, target):
        tmp = torch.sub(prob, target)
        loss = torch.pow(tmp, 2)
        return torch.mean(torch.sum(loss,2))
