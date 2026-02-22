import torch
import torch.nn as nn

torch.device('cpu')
torch.cuda.device('cuda')#第0块GPU
torch.cuda.device('cuda:1')#第1块GPU
torch.cuda.device_count()#返回可用的GPU个数

#以下两个函数允许我们在请求的GPU不存在的情况下运行代码
def try_gpu(i=0):
    #如果存在，返回GPU（i），否则返回CPU
    if torch.cuda.device_count()>= i + 1 :
        return  torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    #返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(),try_gpu(10),try_all_gpus())

#查询张量所在的设备
x = torch.tensor([1,2,3])
print(x.device)#.device获取对象所在设备

#存储在GPU上
X = torch.ones(2,3,device=try_gpu())
print(X)
"""
如果有多张GPU，想在第i个GPU上创建张量，
使用device=try_gpu(i-1)
"""
#张量之间计算时，所有张量必须在同一个GPU上
"""
Z=X.cuda(1)  #Z是X在第二个GPU上的拷贝
Z.cuda(1) is Z #True | 如果往同一块GPU上拷贝，
那么这一步实际上不会执行
"""

#神经网络与GPU
net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())
print(net(X))

#确保模型参数存储在同一个GPU上
print(net[0].weight.data.device)


