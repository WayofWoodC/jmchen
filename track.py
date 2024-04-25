import sys  
import os,time,gc,psutil,sys ,json,pyarrow
import numpy as np
import pandas as pd
import itertools
import multiprocessing
from datetime import datetime,timedelta,date
import warnings
warnings.filterwarnings("ignore")

path_trade_date = '/data/disk3/DataBase_stocks/tradeDates/trade_date.csv'
readpath='/data/disk3/DataBase_stocks/marketData_kuaizhao'
save_dir='/data/disk4/output_stocks/jmchen/factors/shot/track'
#日度因子更新存储路径
save_daily=save_dir+'/daily_files'
#单因子的历史数据
save_factors=save_dir+'/entire_files'

def calc_fun(dfg):
    hprice,lprice=dfg['last_price'].quantile([0.8,0.2])
    htrades_ratio=dfg['num_trades'][dfg['last_price']>hprice].sum()/dfg['num_trades'].sum()
    ltrades_ratio=dfg['num_trades'][dfg['last_price']<lprice].sum()/dfg['num_trades'].sum()
    hvolume_ratio=dfg['volume'][dfg['last_price']>hprice].sum()/dfg['volume'].sum()
    lvolume_ratio=dfg['volume'][dfg['last_price']<lprice].sum()/dfg['volume'].sum()
    htv_ratio=(dfg['volume'][dfg['last_price']>hprice].sum()/dfg['num_trades'][dfg['last_price']>hprice].sum())/(dfg['volume'].sum()/dfg['num_trades'].sum())
    ltv_ratio=(dfg['volume'][dfg['last_price']<lprice].sum()/dfg['num_trades'][dfg['last_price']>hprice].sum())/(dfg['volume'].sum()/dfg['num_trades'].sum())
    # return pd.DataFrame([htrades_ratio,ltrades_ratio,hvolume_ratio,lvolume_ratio,htv_ratiio,ltv_ratiio],columns=['htrades_ratio','ltrades_ratio','hvolume_ratio','lvolume_ratio','htv_ratio','ltv_ratio'])
    result = pd.DataFrame({
        'htrades_ratio': [htrades_ratio],
        'ltrades_ratio': [ltrades_ratio],
        'hvolume_ratio': [hvolume_ratio],
        'lvolume_ratio': [lvolume_ratio],
        'htv_ratio': [htv_ratio],
        'ltv_ratio': [ltv_ratio]
    })
    return result

def daily_factor(trading_day):
    date=trading_day.replace('-','')
    if os.path.exists(save_daily+'/'+date+'.fea'):
        return 
    #读当日快照数据
    print(trading_day)
    df=pd.read_parquet(readpath+'/'+trading_day+'.parquet')
    df=df[(df['trade_time']>=93000000) & (df['trade_time']<=145700000)] #抛去盘前和尾盘
    r=df.groupby('security_id').apply(calc_fun)
    r.index=[x[0][:6] for x in r.index]
    r.to_feather(save_daily+'/'+date+'.fea')
    print('finish')

# ### 全部刷新部分
# start_date=datetime.strptime('20190101','%Y%m%d')
# end_date=datetime.strptime('20191231','%Y%m%d')
# trade_dates = pd.read_csv(path_trade_date)
# trade_dates.Date = pd.to_datetime(trade_dates.Date)
# trade_date = trade_dates[(trade_dates.Date >= start_date) & (trade_dates.Date <= end_date)]
# trade_date.Date = trade_date.Date.astype('str')
# for trading_day in trade_date.Date:
#     daily_factor(trading_day)
# num_processes = 3
# #创建进程池
# args=[trading_day for trading_day in trade_date.Date]
# pool = multiprocessing.Pool(processes=num_processes)
# pool.map(daily_factor,args)

def combine(factors,daily_path=save_daily,factpath=save_factors): 
    files=os.listdir(daily_path)
    files.sort()
    dfs=[]
    for file in files:
        df=pd.read_feather(daily_path+'/'+file)
        df=df.reset_index()
        df.rename(columns={df.columns[0]: 'code'}, inplace=True)
        date=file.split('.')[0]
        df.insert(loc=0, column='date', value=date)
        dfs.append(df)
    df=pd.concat(dfs,axis=0,join='outer')
    for i,factor in enumerate(factors):
        print(factor)
        dfout=df.pivot(index='date', columns='code', values=factor)
        dfout.to_csv(factpath+'/'+factor+'.csv')

factors=['htrades_ratio','ltrades_ratio','hvolume_ratio','lvolume_ratio','htv_ratio','ltv_ratio']
combine(factors=factors)

#直接计算全部因子
# def daily_factor(trading_day):
#     date=trading_day.replace('-','')
#     # if os.path.exists(save_daily+'/'+date+'.fea'):
#     #     return 
#     #读当日快照数据
#     print(trading_day)
#     df=pd.read_parquet(readpath+'/'+trading_day+'.parquet')
#     df=df[(df['trade_time']>=93000000) & (df['trade_time']<=145700000)] #抛去盘前和尾盘
#     r=df.groupby('security_id').apply(calc_fun)
#     r.index=[x[0][:6] for x in r.index]
#     r['trading_day']=date
#     r=r.reset_index()
#     r=r.rename(columns={'index':'code'})
#     print('finish')
#     return r


# def calc_entire(start_date,end_date):
#     trade_dates=pd.read_csv(path_trade_date)
#     trade_dates.Date=pd.to_datetime(trade_dates.Date)
#     trade_date=trade_dates[(trade_dates.Date>=start_date) & (trade_dates.Date<=end_date)]
#     trade_date.Date = trade_date.Date.astype('str')
#     args=[trading_day for trading_day in trade_date.Date]   
#     pool=multiprocessing.Pool(5)
#     results=pool.map(daily_factor,args)
#     pool.close()
#     pool.join()
#     return results

# r=calc_entire(start_date=datetime.strptime('20190101','%Y%m%d'),end_date=datetime.strptime('20211231','%Y%m%d'))
# dfr=pd.concat(r)
# htrades_ratio=dfr.pivot(index='trading_day',columns='code',values='htrades_ratio')
# ltrades_ratio=dfr.pivot(index='trading_day',columns='code',values='ltrades_ratio')
# hvolume_ratio=dfr.pivot(index='trading_day',columns='code',values='hvolume_ratio')
# lvolume_ratio=dfr.pivot(index='trading_day',columns='code',values='lvolume_ratio')
# htv_ratio=dfr.pivot(index='trading_day',columns='code',values='htrades_ratio')
# ltv_ratio=dfr.pivot(index='trading_day',columns='code',values='htrades_ratio')

# htrades_ratio.to_csv(save_factors+'/htrades_ratio.csv')
# ltrades_ratio.to_csv(save_factors+'/ltrades_ratio.csv')
# hvolume_ratio.to_csv(save_factors+'/hvolume_ratio.csv')
# lvolume_ratio.to_csv(save_factors+'/hvolume_ratio.csv')
# htv_ratio.to_csv(save_factors+'/htv_ratio.csv')
# ltv_ratio.to_csv(save_factors+'/ltv_ratio.csv')