#该脚本中的模型核心计算结构都是全连接层
import torch 
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)#设置种子

#简单的多层感知机
net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X =torch.rand(2,20)
print(net(X))

#自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

net=MLP()
print(net(X))

#自定义sequential
class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self,X):
        for block in self._modules.values():
            X=block(X)
        return X

X =torch.rand(2,20)
net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(net(X))

#还可以自定义前向传播函数，获得更大的灵活性
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
    
    def forward(self,X):
        X=self.linear(X)
        X=F.relu(torch.mm(X,self.rand_weight)+1)
        X=self.linear(X)
        while X.abs().sum()>1:
            X/=2
        return X.sum()

X =torch.rand(2,20) 
print(net(X))

#这些块可以自由组合，混合搭配
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))

X =torch.rand(2,20)     
chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print(chimera(X))