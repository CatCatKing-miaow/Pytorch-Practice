#02：这个脚本进行了是否有dropout和WD的两个模型之间的对比。将权重衰退和丢弃法的效果更好地可视化
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 固定随机种子，确保每次运行看到的“噪音点”位置都一样，方便对比
torch.manual_seed(42) 

x_train = torch.linspace(-3, 3, 20).unsqueeze(1)
y_train = torch.sin(x_train) + torch.randn(20, 1) * 0.3  # 真实的 sin(x) + 严重的噪音

x_test = torch.linspace(-3, 3, 100).unsqueeze(1)
y_test = torch.sin(x_test) # 干净的真实规律


#定义一个模型工厂函数 (通过开关控制是否加 Dropout)(搭积木的过程，非常巧妙)

def build_model(use_dropout=False):
    layers = [nn.Linear(1, 100), nn.ReLU()]
    
    # 根据参数决定是否插入 Dropout 层
    if use_dropout:
        layers.append(nn.Dropout(0.2))
        
    layers.extend([nn.Linear(100, 100), nn.ReLU()])
    
    if use_dropout:
        layers.append(nn.Dropout(0.2))
        
    layers.append(nn.Linear(100, 1))
    
    # 使用 *layers 把列表里的层解包放进 Sequential，巧妙的python用法
    return nn.Sequential(*layers)


# 实例化两个模型，形成鲜明对比

# 模型 A (无 Dropout)，优化器 A (无 Weight Decay)
net_overfit = build_model(use_dropout=False)
opt_overfit = torch.optim.Adam(net_overfit.parameters(), lr=0.01, weight_decay=0.0)

# 模型 B (有 Dropout)，优化器 B (加入 Weight Decay)
net_reg = build_model(use_dropout=True)
opt_reg = torch.optim.Adam(net_reg.parameters(), lr=0.01, weight_decay=0.01)

loss_fn = nn.MSELoss()

# ==========================================
# 4. 同步开始训练（赛马机制）
# ==========================================
print("正在同时训练两个模型。。。。")
# 把 epoch 调高到 1000，让模型 A 有充足的时间去“死记硬背”（过拟合更明显）
num_epochs = 1000 

for epoch in range(num_epochs):
    # 两个模型都要开启训练模式
    net_overfit.train()
    net_reg.train()
    
    # --- 训练模型 A  ---
    opt_overfit.zero_grad()
    loss_a = loss_fn(net_overfit(x_train), y_train)
    loss_a.backward()
    opt_overfit.step()
    
    # --- 训练模型 B ---
    opt_reg.zero_grad()
    loss_b = loss_fn(net_reg(x_train), y_train)
    loss_b.backward()
    opt_reg.step()

#结束训练，进入观察模式
net_overfit.eval()
net_reg.eval()

with torch.no_grad():
    y_pred_overfit = net_overfit(x_test)
    y_pred_reg = net_reg(x_test)

# --- 开始画图 ---
plt.figure(figsize=(10, 6))

# 1. 画出底层真实的物理规律 (绿色平滑曲线)
plt.plot(x_test.numpy(), y_test.numpy(), label='True Function (sin(x))', color='green', linewidth=2)

# 2. 画出带噪音的训练数据 (红色散点)
plt.scatter(x_train.numpy(), y_train.numpy(), label='Noisy Train Data', color='red', s=50, zorder=5)

# 3. 画出【模型 A】过拟合的疯狂曲线 (橙色实线)
plt.plot(x_test.numpy(), y_pred_overfit.numpy(), label='Overfitted (No Reg)', color='orange', linestyle='-', linewidth=2)

# 4. 画出【模型 B】被正则化约束的平滑曲线 (蓝色虚线)
plt.plot(x_test.numpy(), y_pred_reg.numpy(), label='Regularized (Dropout + WD)', color='blue', linestyle='--', linewidth=2)

plt.title("The Power of Regularization: Overfitting vs. Regularized Fit")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()