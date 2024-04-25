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

files=os.listdir('/data/disk3/output_stocks/ytcheng/ks_sig_pool_daily')
files.sort()
dfs=[]
for file in files:
    date=file.replace('-','')
    date=int(date.split('.')[0])
    if date<20210101 or date>=20220101:
        continue
    print(date)
    df=pd.read_csv('/data/disk3/output_stocks/ytcheng/ks_sig_pool_daily/'+file,index_col=0)
    # not_nan=df.notna().sum(axis=1)
    # df=df[not_nan>0.8*len(df.columns)]
    df.insert(0,'date',date)
    cols = df.columns.difference(['date'])
    # 使用 astype 方法将这些列更改为 float16 类型
    df[cols] = df[cols].astype(np.float16)
    dfs.append(df)
dfs=pd.concat(dfs,axis=0,join='inner')
dfs.reset_index(inplace=True)
dfs.to_feather('/data/disk4/output_stocks/jmchen/factors/ML/combine21.feather')