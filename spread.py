import sys  
import os,time,gc,psutil,sys ,json,pyarrow
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

path_trade_date = '/data/disk3/DataBase_stocks/tradeDates/trade_date.csv'
readpath='/data/disk3/DataBase_stocks/marketData_kuaizhao'
save_dir='/data/disk4/output_stocks/jmchen/factors/shot/spread'
#日度因子更新存储路径
save_daily=save_dir+'/daily_files'
#单因子的历史数据
save_factors=save_dir+'/entire_files'

# def calc_fun(dfg, w, n):
#     BID=0
#     ASK=0
#     for i in range(0,n):
#         BID+=(dfg['bid%s'%str(i)]/1000)*dfg['bid_volume%s'%str(i)]*(10-i)/10
#         ASK+=(dfg['ask%s'%str(i)]/1000)*dfg['ask_volume%s'%str(i)]*(10-i)/10
#         # BID+=(dfg['bid%s'%str(i)]/1000)*dfg['bid_volume%s'%str(i)]*w[i]
#         # ASK+=(dfg['ask%s'%str(i)]/1000)*dfg['ask_volume%s'%str(i)]*w[i]
#         # 这个因子优化的方向应该会比较多，首先你可以尝试把这个权重调整尝试倒序来加权，
#         # 比如一档位给最小的权重这里权重可以直接换成一个变量方便调整w =[0.1,....]
#         # 然后你选择的n档价格也是可以调整的，比如尝试一下n=5
#         # spread应该会有很多同类型的因子比如这个地方bid，ask还可以调整为vwap的价格，
#         # 计算加权价格的spread，或者是每档价格差的平均值，或者每档的平均价格差。
#     spread=(BID-ASK)/(BID+ASK)
#     spread=spread.mean()
#     return spread

#计算相应档位个数和权重的spread
def calc_fun(dfg, w, n, if_vwap):  #若用vwap则n最好为10
    BID=0
    ASK=0
    for i in range(0,n):
        BID+=(dfg['bid%s'%str(i)]/1000)*dfg['bid_volume%s'%str(i)]*w[i]
        ASK+=(dfg['ask%s'%str(i)]/1000)*dfg['ask_volume%s'%str(i)]*w[i]
    if if_vwap==True:
        BID=BID/dfg['total_bid_volume']
        ASK=ASK/dfg['total_ask_volume']
    spread=(BID-ASK)/(BID+ASK)
    spread=spread.mean()
    return spread

#只看相应档位价格的spread
def calc_fun_price(dfg, w, n): #只看相应档位价格的spread
    BID=0
    ASK=0
    for i in range(0,n):
        BID+=(dfg['bid%s'%str(i)]/1000)*w[i]
        ASK+=(dfg['ask%s'%str(i)]/1000)*w[i]

    spread=(BID-ASK)/(BID+ASK)
    spread=spread.mean()
    return spread

#只看相应档位价差
def calc_fun_price_dif(dfg, w, n): 
    dif=0
    for i in range(0,n):
        dif+=(dfg['ask%s'%str(i)]/1000)-(dfg['bid%s'%str(i)]/1000)
    dif=dif.mean()
    return dif

def daily_factor(trading_day):
    date=trading_day.replace('-','')
    #if os.path.exists(save_daily+'/'+date+'.fea'):
    if os.path.exists('/data/disk4/output_stocks/jmchen/factors/shot/spread/temp_daily_files'+'/'+date+'.fea'):
        return 
    #读当日快照数据
    print(trading_day)
    df=pd.read_parquet(readpath+'/'+trading_day+'.parquet')
    df=df[(df['trade_time']>=93000000) & (df['trade_time']<=145700000)] #抛去盘前和尾盘
    # r_z10_pv=df.groupby('security_id').apply(lambda x: calc_fun(x,[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],10,False))
    # r_n10_pv=df.groupby('security_id').apply(lambda x: calc_fun(x,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],10,False))
    # r_z10_vwap=df.groupby('security_id').apply(lambda x: calc_fun(x,[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],10,True))
    # r_n10_vwap=df.groupby('security_id').apply(lambda x: calc_fun(x,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],10,True))
    # r_a5_pv=df.groupby('security_id').apply(lambda x: calc_fun(x,[0.2,0.2,0.2,0.2,0.2],5,False))
    # r_a5_vwap=df.groupby('security_id').apply(lambda x: calc_fun(x,[0.2,0.2,0.2,0.2,0.2],5,True))
    # r_z10_p=df.groupby('security_id').apply(lambda x: calc_fun_price(x,[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],10))
    # r_n10_p=df.groupby('security_id').apply(lambda x: calc_fun_price(x,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],10))
    # r_a10_p=df.groupby('security_id').apply(lambda x: calc_fun_price(x,[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],10))
    # r_a5_p=df.groupby('security_id').apply(lambda x: calc_fun_price(x,[0.2,0.2,0.2,0.2,0.2],5))
    r_z10_dp=df.groupby('security_id').apply(lambda x: calc_fun_price_dif(x,[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],10))
    r_n10_dp=df.groupby('security_id').apply(lambda x: calc_fun_price_dif(x,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],10))
    r_a10_dp=df.groupby('security_id').apply(lambda x: calc_fun_price_dif(x,[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],10))
    r_a5_dp=df.groupby('security_id').apply(lambda x: calc_fun_price_dif(x,[0.2,0.2,0.2,0.2,0.2],5))
    r=pd.concat([r_z10_dp,r_n10_dp,r_a10_dp,r_a5_dp],axis=1)
    r.columns=['z10_dp','n10_dp','a10_dp','a5_dp']
    # r=pd.concat([r_z10_pv,r_n10_pv,r_z10_vwap,r_n10_vwap,r_a5_pv,r_a5_vwap,r_z10_p,r_n10_p,r_a10_p,r_a5_p,r_z10_dp,r_n10_dp,r_a10_dp,r_a5_dp],axis=1)
    # r.columns=['z10_pv','n10_pv','z10_vwap','n10_vwap','a5_pv','a5_vwap','z10_p','n10_p','a10_p','a5_p','z10_dp','n10_dp','a10_dp','a5_dp']
    r.index=[x[:6] for x in r.index]
    r.to_feather('/data/disk4/output_stocks/jmchen/factors/shot/spread/temp_daily_files'+'/'+date+'.fea')

## 全部刷新部分
start_date=datetime.strptime('20210101','%Y%m%d')
end_date=datetime.strptime('20221231','%Y%m%d')
trade_dates = pd.read_csv(path_trade_date)
trade_dates.Date = pd.to_datetime(trade_dates.Date)
trade_date = trade_dates[(trade_dates.Date >= start_date) & (trade_dates.Date <= end_date)]
trade_date.Date = trade_date.Date.astype('str')
for trading_day in trade_date.Date:
    daily_factor(trading_day)
# num_processes = 3
# #创建进程池
# args=[trading_day for trading_day in trade_date.Date]
# pool = multiprocessing.Pool(processes=num_processes)
# results=pool.map(daily_factor,args)


# def combine(factors,daily_path=save_daily,factpath=save_factors): 
#     files=os.listdir(daily_path)
#     files.sort()
#     dfs=[]
#     for file in files:
#         df=pd.read_feather(daily_path+'/'+file)
#         df=df.reset_index()
#         df.rename(columns={df.columns[0]: 'code'}, inplace=True)
#         date=file.split('.')[0]
#         df.insert(loc=0, column='date', value=date)
#         dfs.append(df)
#     df=pd.concat(dfs,axis=0,join='outer')
#     for i,factor in enumerate(factors):
#         print(factor)
#         dfout=df.pivot(index='date', columns='code', values=factor)
#         dfout.to_csv(factpath+'/'+factor+'.csv')

# factors=['spread']
# combine(factors=factors)

# #直接计算全部因子
# def daily_factor(trading_day):
#     date=trading_day.replace('-','')
#     # if os.path.exists(save_daily+'/'+date+'.fea'):
#     #     return 
#     #读当日快照数据
#     print(trading_day)
#     df=pd.read_parquet(readpath+'/'+trading_day+'.parquet')
#     df=df[(df['trade_time']>=93000000) & (df['trade_time']<=145700000)] #抛去盘前和尾盘
#     r=df.groupby('security_id').apply(calc_fun)
#     r.index=[x[:6] for x in r.index]
#     r=r.reset_index()
#     r['date']=date
#     r=r.rename(columns={'index':'code',0:'spread'})
#     print('finish')
#     return r

def calc_entire(start_date,end_date): #cpu多线程
    trade_dates=pd.read_csv(path_trade_date)
    trade_dates.Date=pd.to_datetime(trade_dates.Date)
    trade_date=trade_dates[(trade_dates.Date>=start_date) & (trade_dates.Date<=end_date)]
    trade_date.Date = trade_date.Date.astype('str')
    args=[trading_day for trading_day in trade_date.Date]   
    pool=multiprocessing.Pool(2)
    results=pool.map(daily_factor,args)
    pool.close()
    pool.join()
    return results

# if __name__ == '__main__':
#     daily_factor('2024-04-12')
    
# r=calc_entire(start_date=datetime.strptime('20190101','%Y%m%d'),end_date=datetime.strptime('20211231','%Y%m%d'))
# dfr=pd.concat(r)
# spread=dfr.pivot(index='date',columns='code',values='spread')
# spread.to_csv(save_factors+'/spread.csv')

# def weight_calc(df):
#     weights=[i for i in range(1,21)]
#     df=df*weights
#     return df.mean()
# df=pd.read_csv('/data/disk4/output_stocks/jmchen/factors/shot/spread/entire_files/spread.csv')
# df=df.set_index('date')
# r=df.rolling(20).apply(weight_calc)