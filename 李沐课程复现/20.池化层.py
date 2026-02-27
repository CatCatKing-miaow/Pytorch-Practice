import torch
import torch.nn as nn

"""最简单的pool2d,padding=0,stride=1"""
def pool2d(X,pool_size,mode='max'):
    p_h,p_w = pool_size
    Y = torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            if mode == 'avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y

X=torch.tensor([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]])

#验证二维最大池化层的输出
print(pool2d(X,(2,2)))
#验证平均池化层
print(pool2d(X,(2,2),'avg'))

"""
    深度学习框架中的步幅与池化窗口的大小相同
    即：kernel_size=stride
"""
X=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
pool2d = nn.MaxPool2d(3)
print(pool2d(X))#输出是10。边角料直接忽略

"""
    填充和步幅可以手动设定。
    还可以设定一个任意大小的矩阵池化窗口，
    分别设定填充和步幅的高度和宽度
"""
pool2d = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))
pool2d = nn.MaxPool2d((2,3),padding=(1,1),stride=(2,3))
print(pool2d(X))

"""池化层在每个输入通道上单独运算"""
X=torch.cat((X,X+1),1)#cat表示合并，第二个参数表示在第一维上合并
print(X)

pool2d = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))