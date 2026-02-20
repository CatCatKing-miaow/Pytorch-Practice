#01：这个脚本实现了基础的Dropout和WD用于防止过拟合
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_train = torch.linspace(-3, 3, 20).unsqueeze(1)  #unsqueeze用于增加维度。unsqueeze(1)表示新增加第一维度（注意不能写2，超范围了）
y_train = torch.sin(x_train) + torch.randn(20, 1) * 0.3  #加噪声

x_test = torch.linspace(-3, 3, 100).unsqueeze(1)         
y_test = torch.sin(x_test)                                #完美的没有噪声

#定义一个具有特别多参数的模型，来应用dropout和wd缓解过拟合


net = nn.Sequential(
    nn.Linear(1, 100),       # 输入层 (1维) -> 隐藏层1 (100维)
    nn.ReLU(),               
    nn.Dropout(0.2),         # dropout：每次随机断开20%的神经元
    
    nn.Linear(100, 100),     # 隐藏层1 (100维) -> 隐藏层2 (100维)
    nn.ReLU(),
    nn.Dropout(0.2),         # dropout：依旧20%
    
    nn.Linear(100, 1)        # 隐藏层2 -> 输出层 (回归问题，输出1维连续值，不加激活函数)
    )




loss_fn = nn.MSELoss() # 回归问题常用的均方误差

# 在优化器中加入 weight_decay (L2正则化)
# 这里设置 wd=0.01，强迫模型在更新时不断压缩权重的绝对值，让拟合的曲线变得平滑
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.01)

num_epochs = 500
for epoch in range(num_epochs):
    net.train() # 开启训练模式，Dropout 生效

    # 前向传播
    y_hat = net(x_train)
    loss = loss_fn(y_hat, y_train)
    
    # 经典反向传播三步曲
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # 这一步里，权重因为 weight_decay 缩水了一点点，所以叫做权重衰退


net.eval() # 切回测试模式
with torch.no_grad(): # 关闭梯度记录
    # 让模型预测测试集那 100 个点
    y_pred = net(x_test)

# 使用 matplotlib 将结果可视化

plt.figure(figsize=(8, 5))
# 画出真实的无噪音规律（平滑的正弦曲线）
plt.plot(x_test.numpy(), y_test.numpy(), label='True function (sin(x))', color='green', linewidth=2)
# 画出带有噪音的训练数据点
plt.scatter(x_train.numpy(), y_train.numpy(), label='Noisy Train Data', color='red', s=40)
# 画出模型最终拟合出的曲线
plt.plot(x_test.numpy(), y_pred.numpy(), label='Model Prediction (with Dropout & WD)', color='blue', linestyle='--')

plt.title("MLP Regression with Weight Decay and Dropout")
plt.legend()
plt.show()