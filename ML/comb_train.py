import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)
else:
    device = torch.device("cpu")
    print("GPU 不可用，将在 CPU 上运行")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()  # 释放显存

# 定义标准化函数
def normalize_row(row):
    data = row.dropna()
    mean = np.mean(data)
    std = np.std(data)
    row[row.notna()] = (row[row.notna()] - mean) / std
    return row


#因子正交
class MyModelZJ(nn.Module):
    def __init__(self, input_size,num_factors, num_stocks):
        super(MyModelZJ, self).__init__()
        self.num_factors = num_factors
        self.num_stocks = num_stocks
               
        # 定义全连接层
        self.fc = nn.Linear(input_size, num_factors)  #
        
        # 定义批标准化层
        self.bn = nn.BatchNorm1d(num_factors)  #

    def forward(self, x):
        batch_size, _, num_features = x.size()
        x = x.view(-1, num_features)
        # 全连接层
        fc_out = self.fc(x)
        fc_out=fc_out.view(batch_size*self.num_stocks,self.num_factors)
        factor_output = self.bn(fc_out)
        factor_output = factor_output.view(batch_size, self.num_stocks, self.num_factors)
        return factor_output
def custom_loss(y, factor_output,k):  #惩罚项系数k
    c = factor_output.mean(dim=2)
    corr=[]
    for i in range(c.shape[0]): #batch内循环
        mean_c = torch.mean(c[i])
        mean_y = torch.mean(y[i])
        
        # Calculate the numerator and denominators for Pearson correlation
        numerator = torch.sum((c[i] - mean_c) * (y[i] - mean_y))
        denominator_c = torch.sqrt(torch.sum((c[i] - mean_c)**2))
        denominator_y = torch.sqrt(torch.sum((y[i] - mean_y)**2))
        # Calculate the Pearson correlation coefficient
        pearson_corr = numerator / (denominator_c * denominator_y)
        corr.append(pearson_corr)
    corr=torch.stack(corr).mean()
    penalty=[]
    for i in range(factor_output.shape[0]): #batch内循环
        correlation_matrix = torch.corrcoef(factor_output[i].T)
        # 将对角线元素设置为零，即将每列与自身的相关系数剔除
        for i in range(factor_output.shape[2]):
            correlation_matrix[i, i] = 0
        # 计算相关系数矩阵的L2范数
        zsum = torch.sum(correlation_matrix**2)
        penalty.append(zsum)
    penalty=torch.stack(penalty).mean().sqrt()
    loss=-corr+k*penalty
    return loss

df20=pd.read_feather('/data/disk4/output_stocks/jmchen/factors/ML/combine20.feather')
df21=pd.read_feather('/data/disk4/output_stocks/jmchen/factors/ML/combine21.feather')
df=pd.concat([df20,df21],join='inner',axis=0)
df=df.drop('level_0',axis=1)
index_counts = df['index'].value_counts()
index_counts=index_counts[index_counts==index_counts.max()]
print(index_counts)
dfx=df[df['index'].isin(index_counts.index)]
dfx=dfx.fillna(0)
ds=dfx.groupby('tradingday')
days=index_counts.max()
num_stocks=3002
num_features=4370
X = torch.zeros((days, num_stocks, num_features))
Y = torch.zeros((days, num_stocks))
i=0
day_list=[]
code_list=[]
for day,df in ds:
    print(day,i)
    if i==1:
        code_list=df['index'].to_list()
    df=df.set_index('index')
    x= df.drop(columns=['tradingday','ret'])
    X[i]=torch.Tensor(x.values)
    y=df['ret']
    Y[i]=torch.Tensor(y.values)

    i+=1
    if i>=400:
        day_list.append(day)

X_train=X[:400]
X_test=X[400:]
Y_train=Y[:400]
Y_test=Y[400:]


from torch.optim import lr_scheduler
from itertools import product


def train_and_save_model(X_train, Y_train, num_factors, lr, batch_size, k, num_epochs):
    print( num_factors, lr, batch_size, k)
    input_size = num_features
    model = MyModelZJ(input_size, num_factors, num_stocks)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=1)
    
    loss_history = []
    count = 0
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        loss_epoch = []
        for i in range(0, X_train.size(0) - batch_size, batch_size):
            inputs = X_train[i:i + batch_size].to(device)
            labels = Y_train[i:i + batch_size].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(labels, outputs, k)
            loss_epoch.append(loss.item() / num_stocks)
            loss.backward()
            optimizer.step()
        
        loss_history.append(np.mean(loss_epoch))
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} loss={np.mean(loss_epoch)} - Learning Rate: {current_lr}")
        
        if epoch >= 3:
            if (loss.item() / num_stocks) > last:
                count += 1
                if count >= 3:
                    print('不收敛')
                    break
        last = loss.item() / num_stocks

    # 保存模型
    model_filename = f"model_num_factors_{num_factors}_lr_{lr}_batch_{batch_size}_k_{k}.pt"
    model_filepath = os.path.join("/data/disk4/output_stocks/jmchen/factors/ML/zj_model", model_filename)
    torch.save(model.state_dict(), model_filepath)
    torch.cuda.empty_cache()
    return np.mean(loss_epoch)

# 定义超参数的搜索范围
num_factors_list = [300, 400, 500]
lr_list = [0.001, 0.01, 0.1]
batch_size_list = [4, 8]
k_list = [0.01, 0.1, 1.0]
num_epochs_list = [120]

# 使用嵌套的列表推导式遍历超参数组合
for num_factors, lr, batch_size, k, num_epochs in product(num_factors_list, lr_list, batch_size_list, k_list, num_epochs_list):
    train_and_save_model(X_train, Y_train, num_factors, lr, batch_size, k, num_epochs)