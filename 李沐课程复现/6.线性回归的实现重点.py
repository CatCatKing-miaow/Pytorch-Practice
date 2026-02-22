import torch

def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))#手动reshape防止广播机制隐患

true_w = torch.tensor([2,-3.4])#w是一维向量，但是matmul能够智能处理
true_b = 4.2
features,labels=synthetic_data(true_w,true_b,1000)#获取带标签的数据