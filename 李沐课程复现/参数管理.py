import torch
import torch.nn as nn

torch.manual_seed(42)

#具有单隐藏层的MLP
net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X=torch.rand(size=(2,4))
print(net(X))

#参数访问
print(net[2].state_dict())

#访问某个具体的参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

"""
net.named_parameters():这是一个迭代器，它会返回该层中所有可学习参数的名字和对应的张量。
对于全连接层，参数通常是 weight 和 bias。
命名规则: 在 nn.Sequential 中，PyTorch 会自动给各层编号。
第一层编号为 0，第二层 ReLU 编号为 1（但 ReLU 没有参数），第三层编号为 2。
输出示例: ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) 
('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
注意：你会看到参数名带上了前缀 0. 和 2.。
"""
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])

"""
.state_dict(): 它返回一个 Python 字典，把网络中每一层的所有参数映射到对应的张量上。
它通常用于保存和加载模型模型。
['2.bias']: 通过键访问字典里的值。2.bias 代表第 3 层（索引为 2 的那层）的偏置项。
.data: 从参数对象中提取纯粹的张量数据。它只包含数值，不包含梯度等额外信息。
"""
print(net.state_dict()['2.bias'].data)

#从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())    
        """
        net.add_module(name, module)
        参数 name (字符串)：为你添加的这一层起一个名字。在代码中使用了 f-string，
        所以产生的名字是 'block0', 'block1', 'block2', 'block3'。
        module (nn.Module对象)：要添加的具体子模块。这里调用了 block1() 函数，
        它会返回一个 nn.Sequential 实例。
        """
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1))
print(rgnet)

#内置的初始化

def init_normal(m):#正态分布初始化
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
print(net[0].weight.data,net[0].bias.data)

def init_constant(m):#初始化为常数（不建议使用）
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data,net[0].bias.data)



#对某些块可以应用不同的初始化方法
def xavier(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data)
print(net[2].weight.data)

#自定义初始化
def my_init(m):
    if type(m)==nn.Linear:#这里的个性化初始化很不常见，这里只是为了方便示例
        print("Init",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])

net[0].weight.data[:]+=1   #直接对数据进行操作也是可行的


#参数绑定（某些层共享参数）
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
X=net(X)
print(net[2].weight.data==net[4].weight.data)#可以看到第二层和第三层参数相等
net[2].weight.data[0,0]=100
print(net[2].weight.data==net[4].weight.data)#改变第二层，第三层也改变
