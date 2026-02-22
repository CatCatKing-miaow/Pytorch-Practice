#检验不同初始化对信号强度的影响
import torch
import torch.nn as nn

torch.manual_seed(42)#固定随机种子

def simulate_signal_flow(init_type,activation_type,num_layers=30,hidden_size=512):
    """
    模拟信号穿过30层神经网络的过程，并且追踪方差(信号强度)
    """
    print(f"==========实验：{init_type}+{activation_type}==========")
    x=torch.randn(1000,hidden_size)#输入信号：1000个样本，512个特征，标准正态分布，初始方差接近1
    print(f"初始输入信号的方差：{x.var().item():.4f}")

    for i in range(num_layers):
        #定义一个纯线性层（为了单纯观察权重的影响，我们关掉bias）
        layer = nn.Linear(hidden_size,hidden_size,bias=False)
        if init_type=="naive":
            nn.init.normal_(layer.weight,mean=0,std=0.01)
        elif init_type=="xavier":
            nn.init.xavier_normal_(layer.weight)
        elif init_type =="He":
            nn.init.kaiming_normal_(layer.weight,nonlinearity="relu")
        
        x=layer(x)

        if activation_type=="Tanh":
            x=torch.tanh(x)
        elif activation_type=="ReLU":
            x=torch.relu(x)
        
        if(i+1)%10==0:
            print(f"穿过第{i+1}层后，信号方差：{x.var().item():.8f}")
    print("\n")

if __name__ =="__main__":
    #e.g.1   Xavier and Tanh
    #1.1 no Xavier
    simulate_signal_flow("naive","Tanh")
    #1.2 with Xavier
    simulate_signal_flow("xavier","Tanh")
    #e.g.2   He and ReLU
    #2.1    不正确的搭配： xavier和ReLU
    simulate_signal_flow("xavier","ReLU")
    #2.2    正确搭配：He 和ReLU
    simulate_signal_flow("He","ReLU")


#一开始，我纠结于为什么经过30层后，
# 即使使用恰当的初始化方法，还是损失高达60%。在和Gemini的沟通中，我受到了启发：
"""
残酷现实 A：分布形状的“悄悄变形”
何恺明的推导中有一个隐藏前提：输入到 ReLU 之前的信号，必须是一个完
美的正态分布（钟形曲线），且均值为 0。 只有在这种完美对称的情况下，
ReLU 才刚好极其精准地切掉绝对的 50% 方差。
但在代码中，经过第一层 ReLU 后，所有负数被强行变成了 0。这就导致下
一层的输入虽然经过了线性层的混合（重新把均值拉回了 0 附近），但它的
概率分布形状其实已经微微偏离了完美的正态分布。当形状不再完美对称时，
ReLU 砍掉的方差就不再是极其精准的 50%（可能会变成 52% 或 53%）。这
微小的误差，在 30 层的累积下，就变成了你看到的从 1.0 掉到 0.39。

残酷现实 B：有限宽度的随机性（抽样误差）
数学推导是基于“无穷大”的维度的。我们在代码里设置的特征维度 hidden_size=512
虽然不小，但在统计学上依然是有限的。每次用正态分布生成 512×512 的权重矩阵时，
实际生成的方差并不会不多不少正好等于理论值，会有微小的随机波动。

为了彻底、百分之百地锁死方差，大佬们（比如何恺明后来的 ResNet）不再仅仅依赖
“初始化”这一锤子买卖，而是引入了深度学习的另一个极其伟大的发明：归一化层（Normalization，
比如 BatchNorm 或 LayerNorm）。

它的逻辑无比霸道：既然不管怎么精妙地初始化，跑几十层之后方差还是会慢慢漂移，
那我就不在初始化上死磕了！ 我直接在每一层的 ReLU 前面加一个“保安”，强行把
当前数据的均值重新拉回 0，方差重新强行放大回 1.0！
"""