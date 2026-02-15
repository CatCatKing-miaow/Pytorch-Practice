import torch
#核心内容：针对不同的axis求和；keepdims的使用（便于广播）
a=torch.ones(2,5,4)
print(a.sum(axis=1).shape)
print(a.sum(axis=0).shape)
print(a.sum(axis=[0,2]).shape)
print(a.sum(axis=[0,2],keepdims=True).shape)