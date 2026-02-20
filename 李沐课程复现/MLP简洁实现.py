import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

trans = transforms.ToTensor()

mnist_train=torchvision.datasets.FashionMNIST(root="./data(MLP)",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="./data(MLP)",train=False,transform=trans,download=True)
train_iter = DataLoader(mnist_train,batch_size=256,shuffle=True)
test_iter = DataLoader(mnist_test,batch_size=256,shuffle=False)

net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def ea(net,data_iter):#利用测试集计算准确率
    net.eval()

    acc_sum=0.0
    total_samples = 0

    with torch.no_grad():
        for X,y in data_iter:
            y_hat=net(X)
            acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            total_samples+=y.size(0)
    return acc_sum/total_samples

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_weights)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 10
print("开始训练！")

for epoch in range(num_epochs):
    train_loss_sum=0.0
    train_acc_sum=0.0
    total_samples=0.0
    net.train()

    for X,y in train_iter:
        y_hat=net(X)
        l = loss(y_hat,y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss_sum += l.item() * y.size(0)
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()#argmax找第一维的最大值，随后比较，求和。
        total_samples += y.size(0)
        
    avg_loss = train_loss_sum / total_samples
    avg_acc = train_acc_sum / total_samples
    test_acc=ea(net,test_iter)
    
    print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {avg_acc:.4f},Test acc={test_acc:.4f}")