import torch
import torch.nn as nn
import torch.nn.functional as F


#构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return X - X.mean()#通过这一步，E=0

layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

#可以将层作为组件构建更复杂的模型。（仍是nn.Module的子类）
net=nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y=net(torch.rand(4,8))
print(Y.mean().item())#均值为0（输出为极小的实数，因为存在浮点数误差）

#带参数的图层
class MyLinear(nn.Module):#自定义一个线性层
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))#自定义参数要放到nn.Parameter类中
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self,X):
        linear = torch.matmul(X,self.weight.data)+self.bias.data
        return F.relu(linear)
    
dense = MyLinear(5,3)
print(dense.weight)

#使用自定义层直接进行正向传播计算
dense(torch.rand(2,5))

#使用自定义层构建模型
net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print(net(torch.randn(2,64)))

