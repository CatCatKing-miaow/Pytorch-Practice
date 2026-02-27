import torch
from torch import nn
def corr2d(X,K):#和17中的相同
    """计算二维互相关运算。"""
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

"""实现多输入通道互相关运算"""
def corr2d_multi_in(X,K):
    return sum(corr2d(x,k) for x,k in zip(X,K))

"""计算多通道输出的互相关函数"""
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

"""1*1卷积"""
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X = X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y = torch.matmul(K,X)
    return Y.reshape((c_o,h,w))

X=torch.normal(0,1,(3,3,3))
K = torch.normal(0,1,(2,3,1,1))
Y1 = corr2d_multi_in_out_1x1(X,K)
Y2 = corr2d_multi_in_out(X,K)
assert float(torch.abs(Y1-Y2).sum()) <1e-6