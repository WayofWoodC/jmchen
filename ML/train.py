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
from scipy.stats import spearmanr
from scipy.stats import pearsonr

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
#数据处理
print('读取数据')
df=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/adjopen.csv',index_col=0)
df.drop([x for x in df.columns if x[-2:]=='BJ'],axis=1,inplace=True)
dfopen = df.apply(normalize_row, axis=1)
df=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/adjclose.csv',index_col=0)
df.drop([x for x in df.columns if x[-2:]=='BJ'],axis=1,inplace=True)
dfclose = df.apply(normalize_row, axis=1)
df=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/adjhigh.csv',index_col=0)
df.drop([x for x in df.columns if x[-2:]=='BJ'],axis=1,inplace=True)
dfhigh = df.apply(normalize_row, axis=1)
df=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/adjlow.csv',index_col=0)
df.drop([x for x in df.columns if x[-2:]=='BJ'],axis=1,inplace=True)
dflow = df.apply(normalize_row, axis=1)
df=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/volume.csv',index_col=0)
df.drop([x for x in df.columns if x[-2:]=='BJ'],axis=1,inplace=True)
dfvolume = df.apply(normalize_row, axis=1)
df=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/vwap_adj.csv',index_col=0)
df.drop([x for x in df.columns if x[-2:]=='BJ'],axis=1,inplace=True)
dfvwap = df.apply(normalize_row, axis=1)
dfs=[dfclose,dfopen,dfhigh,dflow,dfvwap,dfvolume]
print('完成标准化')
dff=[]
for df in dfs:  #删除训练集时间段内未上市的
    df_filled = df[(df.index < 20210101) & (df.index>20100101)]
    df_filled=df_filled.dropna(axis=1,how='all')
    df_filled = df_filled.fillna(method='bfill')
    dff.append(df_filled)

dfm=pd.concat(dff,axis=0,join='inner')
columns_to_drop = []
for column in dfm.columns:
    # 将每列转换为布尔值，True表示空值，False表示非空值
    is_null = dfm[column].isnull()
    # 使用rolling函数计算连续空值的长度
    consecutive_nulls = is_null.rolling(5, min_periods=1).sum()
    # 检查是否存在连续空值长度大于等于5的情况
    if any(consecutive_nulls >= 5):
        columns_to_drop.append(column)
for i in range(len(dff)):
    dff[i] = dff[i].drop(columns=columns_to_drop)
#再对齐一下列，不知道为什么concat后还是没对齐：
dfmm=pd.concat(dff,axis=0,join='inner')
for i in range(len(dff)):
    dff[i] = dff[i][dff[i].columns.intersection(dfmm.columns)]
time_length =2674
stock_code_length = 4121
T=60
# 创建一个空的三维数组，用于存放合并后的数据
X_tensor = np.zeros((time_length, stock_code_length, 6))

# 合并6个DataFrame的数据
for i, df in enumerate(dff):
    # 将DataFrame的值复制到对应的tensor切片中
    X_tensor[:, :, i] = df.values
X_tensor=torch.Tensor(X_tensor)
total_samples = 2615  # 指定总样本数量
time_steps_per_sample = 60  # 指定每个样本的时间步数
num_stocks = 4121  # 股票数量
num_features = 6  # 特征数量
# 初始化新的X_train
X_train = torch.zeros((total_samples, time_steps_per_sample, num_stocks, num_features))
# 将数据拆分成样本
for i in range(total_samples):
    end_idx = i + time_steps_per_sample
    X_train[i] = X_tensor[i:end_idx]
print('打标签')
close=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/adjclose.csv',index_col=0)
close.drop([x for x in close.columns if x[-2:]=='BJ'],axis=1,inplace=True)
ret=(close-close.shift(1))/close.shift(1)
ret= ret.fillna(method='bfill')
ret = ret.apply(normalize_row, axis=1) #标准化
ret=ret[(ret.index <=20210104) & (ret.index>=20100101)]
ret= ret[ret.columns.intersection(dff[0].columns)]
ret=ret.iloc[T:,]
array = ret.to_numpy()  # 将DataFrame转换为NumPy数组
Y_train = np.reshape(array, ret.shape)  # 将数组reshape为张量
Y_train=torch.Tensor(Y_train)
print('测试集')
dfftest=[]
for df in dfs:  #删除测试集时间段内未上市的
    df_filled = df[(df.index < 20211231) & (df.index>=20210101)]
    df_filled=df_filled.dropna(axis=1,how='all')
    df_filled = df_filled.fillna(method='bfill')
    dfftest.append(df_filled)
for i in range(len(dff)):
    dfftest[i] = dfftest[i][dfftest[i].columns.intersection(dfmm.columns)]
print(dfftest[0].shape)
time_length =242
stock_code_length = 4121
T=60
# 创建一个空的三维数组，用于存放合并后的数据
X_tensortest = np.zeros((time_length, stock_code_length, 6))

# 合并6个DataFrame的数据
for i, df in enumerate(dfftest):
    # 将DataFrame的值复制到对应的tensor切片中
    X_tensortest[:, :, i] = df.values
X_tensortest=torch.Tensor(X_tensortest)
total_samples = 183  # 指定总样本数量
time_steps_per_sample = 60  # 指定每个样本的时间步数
num_stocks = 4121  # 股票数量
num_features = 6  # 特征数量

# 初始化新的X_train
X_test1 = torch.zeros((total_samples, time_steps_per_sample, num_stocks, num_features))

# 将数据拆分成样本

for i in range(total_samples):
    end_idx = i + time_steps_per_sample
    X_test1[i] = X_tensortest[i:end_idx]

close=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/adjclose.csv',index_col=0)
close.drop([x for x in close.columns if x[-2:]=='BJ'],axis=1,inplace=True)
ret=(close-close.shift(1))/close.shift(1)
ret= ret.fillna(method='bfill')
ret=ret[(ret.index <= 20211231) & (ret.index>=20210101)]
ret= ret[ret.columns.intersection(dff[0].columns)]
ret=ret.iloc[T:,]
array = ret.to_numpy()  # 将DataFrame转换为NumPy数组
Y_test1 = np.reshape(array, ret.shape)  # 将数组reshape为张量
Y_test1=torch.Tensor(Y_test1)

#多因子预测
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_factors, num_stocks):
        super(MyModel, self).__init__()
        self.num_factors = num_factors
        self.num_stocks = num_stocks
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,batch_first=True)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, num_factors)  #h
        
        # 定义批标准化层
        self.bn = nn.BatchNorm1d(num_factors)  #z

    def forward(self, x):
        batch_size, T, _, num_features = x.size()
        
        # 将num_stocks维度移到batch_size之后
        x = x.view(-1, T, num_features)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        fc_out = self.fc(lstm_out)
        
        # 批标准化层
        factor_output = self.bn(fc_out)
        
        # 将结果恢复成(batch_size, num_stocks, num_factors)形状
        factor_output = factor_output.view(batch_size, self.num_stocks, self.num_factors)
        
        # 计算c，这里直接求因子平均
        c = factor_output.mean(dim=2)
        
        return factor_output

def custom_loss(y, factor_output):
    c = factor_output.mean(dim=2)
    corr=[]
    for i in range(c.shape[0]): #batch内循环
        # # 计算秩次差值
        # rank_X = torch.argsort(c[i].reshape(-1))
        # rank_Y = torch.argsort(y[i].reshape(-1))
        # differences = rank_X - rank_Y
        # # 计算斯皮尔曼秩相关系数
        # n = len(c[i])
        # spearman_corr = 1 - (6 * torch.sum(differences**2)) / (n * (n**2 - 1))
        # corr.append(spearman_corr)
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
        # 计算相关系数的平方和
        zsum = torch.sum(correlation_matrix**2)
        penalty.append(zsum)
    penalty=torch.stack(penalty).mean()
    loss=-corr+penalty
    return loss

torch.cuda.empty_cache()  # 释放显存
torch.cuda.empty_cache()
torch.cuda.empty_cache()

print('加载保存的模型参数')
#input_size, hidden_size, num_factors, num_stocks
model = MyModel(6,64,60,4121)
# 加载保存的模型参数
model.load_state_dict(torch.load('/data/disk4/output_stocks/jmchen/factors/ML/model_params_long2.pth'))

# 设置模型为评估模式
model.eval()
ic_values = []

with torch.no_grad():
    input_sample = X_test1  # 获取单个样本并添加批次维度
    label_sample = Y_test1
    # 使用模型进行推理并获取单个样本的预测输出（c值）
    factors = model(input_sample)
    c_value=factors.mean(dim=2)
    print(c_value.shape)
    print(label_sample.shape)
    corr1=[]
    corr2=[]
    for i in range(c_value.shape[0]): 
        # 创建遮罩以识别c_value和Y_test中的非NaN值
        mask_c = ~torch.isnan(c_value[i])
        mask_y = ~torch.isnan(label_sample[i])
        
        # 组合这些遮罩以获取c_value和Y_test的共同遮罩
        mask = mask_c & mask_y

        # 将遮罩应用于张量
        c_value_filtered = c_value[i][mask]
        label_sample_filtered = label_sample[i][mask]

    
        # 计算每个输入的均值
        mean_c = torch.mean(c_value_filtered)
        mean_y = torch.mean(label_sample_filtered)

        # 计算皮尔逊相关系数的分子和分母
        numerator = torch.sum((c_value_filtered - mean_c) * (label_sample_filtered - mean_y))
        denominator_c = torch.sqrt(torch.sum((c_value_filtered - mean_c)**2))
        denominator_y = torch.sqrt(torch.sum((label_sample_filtered - mean_y)**2))

        # 计算皮尔逊相关系数
        pearson_corr = numerator / (denominator_c * denominator_y)
        corr1.append(pearson_corr.item())
        # 计算秩次差值
        rank_X = torch.argsort(c_value[i].reshape(-1))
        rank_Y = torch.argsort(label_sample[i].reshape(-1))
        differences = rank_X - rank_Y
        # 计算斯皮尔曼秩相关系数
        n = len(c_value[i])
        spearman_corr = 1 - (6 * torch.sum(differences**2)) / (n * (n**2 - 1))
        corr2.append(spearman_corr)
    corr1=np.mean(corr1)
    corr2=np.mean(corr2)
    print('Pearson:',corr1)
    print('Spearman:',corr2)
