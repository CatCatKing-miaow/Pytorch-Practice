#该模型中利用sklearn加载数据，RMSE显示误差
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("load data...")
data = fetch_california_housing()
X,y = data.data,data.target#解包。X代表包含所有特征的矩阵，y代表真实房价的向量

#train_test_split随机分割数据。0.2代表测试集的占比
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler() #Z-Score 标准化

#fit（拟合/计算）： 它的任务是“摸底”。它会去遍历你的数据，计算出这批数据的平均值  和 标准差 ，
# 然后悄悄把这两个数字记在 scaler 自己的肚子里。它只计算，不修改原数据。
# transform（转换）： 它的任务是“干活”。它会掏出肚子里记下的 $\mu$ 和 $\sigma$，
# 对你传给它的数据执行真正的标准化公式：$z = \frac{x - \mu}{\sigma}$。
# 它输出缩放后的新数据。
X_train_scaled=scaler.fit_transform(X_train)#算出 μ 和 σ 保存在scaler中，并标准化训练集
X_test_scaled=scaler.transform(X_test)#直接用之前保存的μ 和 σ标准化测试集

#转化数据为pytorch张量
X_train_tensor=torch.tensor(X_train_scaled,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train,dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test_scaled,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test,dtype=torch.float32).view(-1,1)

batch_size=128
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)#对齐，合并数据
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

model = nn.Linear(in_features = 8,out_features=1)
criterion = nn.MSELoss()

#RMSE：计算相对误差，最后输出在终端的也是相对误差。
def log_rmse(net,features,labels):
    clipped_preds = torch.clamp(net(features),0.15,float('inf'))
    rmse = torch.sqrt(criterion(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-3)

epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_X,batch_y in train_loader:
        optimizer.zero_grad()
        predictions=model(batch_X)
        loss = criterion(predictions,batch_y)
        loss.backward()
        optimizer.step()

    if(epoch+1)%10==0:
        model.eval()
        with torch.no_grad():
            train_err = log_rmse(model,X_train_tensor,y_train_tensor)
            test_err  = log_rmse(model,X_test_tensor,y_test_tensor)
        print(f"Epoch{epoch+1:03d}|Log RMSE|training set:{train_err:.4f}|testing set:{test_err:.4f}")
