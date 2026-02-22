import torch
x= torch.arange(4.0)
x.requires_grad_(True)
y=2*torch.dot(x,x)

y.backward()
x.grad.zero_()   #梯度清零
y=x*x            #哈达玛积
u=y.detach()     #把u赋值为常数，值为y
z=u*x
z.sum().backward()#不直接对z反向传播，不然会鼓捣出来一个雅可比矩阵
print(x.grad)