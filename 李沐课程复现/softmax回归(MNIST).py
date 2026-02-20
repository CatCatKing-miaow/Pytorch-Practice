import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data(softmax)',train=True,transform=trans,download=True)
train_iter=torch.utils.data.DataLoader(mnist_train,batch_size=256,shuffle=True)

net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = optim.SGD(net.parameters(),lr=0.1)
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0.0
    total_samples = 0

    for x,y in train_iter:
        trainer.zero_grad()
        y_hat=net(x)
        l=loss(y_hat,y)
        l.backward()
        trainer.step()
        total_loss +=l.item()*len(y)
        total_samples+=len(y)

    print(f"第{epoch+1}轮,平均loss：{total_loss/total_samples:.4f}")

