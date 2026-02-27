import torch
from torch import nn

"""在所有侧边填充一个像素"""
def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.reshape(Y.shape[2: ])

conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)#padding表示填充
X=torch.rand(size=(8,8))
print(comp_conv2d(conv2d,X).shape)

"""填充不同的高度和宽度"""
conv2d = nn.Conv2d(1,1,kernel_size=(5,3),padding = (2,1))
print(comp_conv2d(conv2d,X).shape)

"""将高度和宽度的步幅设置为2"""
conv2d=nn.Conv2d(1,1,kernel_size = 3,padding=1,stride=2)
print(comp_conv2d(conv2d,X).shape)

"""一个更复杂的例子"""
conv2d = nn.Conv2d(1,1,kernel_size=(3,5),padding =(0,1),stride=(3,4))
print(comp_conv2d(conv2d,X).shape)

"""总结：

    output = [ (input + 2 × padding - kernel_size) / stride ] + 1

"""