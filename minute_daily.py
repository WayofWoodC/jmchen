import numpy as np
import pandas as pd
import math
import multiprocessing
import functools
import numpy_ext as npext
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import acorr_ljungbox
import os,sys
from datetime import datetime,timedelta,date
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/jmchen/files")
from demo_email import send_mail
from demo_email import receivers

path_trade_date = '/data/disk3/DataBase_stocks/tradeDates/trade_date.csv'
readpath='/data/disk3/DataBase_stocks/riceQuantData/minute_price/CS/'
#因子存储根目录
save_dir='/data/disk4/output_stocks/jmchen/factors/minutes'
#日度因子更新存储路径
save_daily=save_dir+'/daily_files'
save_roll_daily=save_dir+'/daily_roll_files'
#单因子的历史数据
save_factors=save_dir+'/entire_files'
#中间变量日度存储路径
save_mid=save_dir+'/mid_files'
save_mid_entire=save_dir+'/mid_entire_files'
#因子rolling天数
rolling_dic={'ret_imp':1,'TTV':1,'com_dw':15,'com_LBQ':15,'highret':1,'uid':1,'mid_JAR':20,'mid_TSRJVP':20}

total_amount=pd.read_csv('/data/disk3/DataBase_stocks/AllSample/freeshareValue.csv')
total_amount.columns=[x[:6] for x in total_amount.columns]
total_amount.set_index('Unname',inplace=True)
total_amount = total_amount[[col for col in total_amount.columns if not col.startswith('8')]]

total_shares= pd.read_csv('/data/disk3/DataBase_stocks/AllSample/float_shares.csv', index_col=0)
total_shares.index.name='date'
total_shares.columns=[x[:6] for x in total_shares.columns]
total_shares.drop([x for x in total_shares.columns if x[0]=='8'], axis=1,inplace=True)

mu1=math.sqrt(2)/math.gamma(0.5)
mu2=2**(2/3)*math.gamma(7/6)/math.gamma(0.5)
mu23=2**(1/3)*math.gamma(5/6)/math.gamma(0.5)
mu6=8*math.gamma(7/2)/math.gamma(0.5)
mu32=2**(3/4)*math.gamma(5/4)/math.gamma(0.5)
phi=0.8289

def prod(df):
    p=df.prod()
    return abs(p)
def prodt(df):
    p=df.prod()**(4/3)
    return abs(p)
def prodi(df):
    p=df.prod()**(2/3)
    return abs(p)
def prodo(df):
    p=abs(df.prod())**(3/2)
    return p
def emul_and_sum(df1,df2):
    result=np.sum(df1*df2)
    result=math.exp(result)-1
    return result

### 各个因子计算函数
#改进跳跃因子相关
def calc_ITIV(trading_day,df): 
    df['order_book_id']=df['order_book_id'].str[:6]
    df=df[~df['order_book_id'].str.startswith('8')]
    groups=df[['order_book_id','trade_time','close']].groupby('order_book_id')
    lc=[]
    lTSRJVP=[]
    lJAR=[]
    for code, group in groups:
        ret=group.iloc[-1]['close']/group.iloc[0]['close']-1
        
        group['easy']=group['close']
        group['easy']=(group['easy']-group['easy'].shift(1))/group['easy'].shift(1)  #简单收益率
        group['close']=np.log(group['close'])
        group['close']=(group['close']-group['close'].shift(1)) #对数收益率
        condition=group['close']>0
        reti=condition.astype(int)  #每分钟是否正收益
        RVp=np.sum(group['close']*group['close']*reti)

        IV=np.sum(group['close'].rolling(3).apply(prodi))
        IV=IV*mu23**(-3)

        Rv=np.sum(group['close']*group['close'])
        Bv=np.sum(group['close'].rolling(2).apply(prod))
        Bv=mu1**(-2)*240/239*Bv
        Tp=np.sum(group['close'].rolling(3).apply(prodt))
        Tp=Tp*mu2**(-3)*240*240/238
        if Bv != 0 and Rv != 0:
            t1=((math.pi/2)**2+math.pi-5)/240*np.max([1,Tp/(Bv**2)])
            T=(1-Bv/Rv)/np.sqrt(t1)  #############
        else:
            T=None
        
        #I_BNS
        if T==None:
            i=None
        elif T>1/phi:
            i=1
        else: i=0
        #𝐼𝐽𝑢𝑚𝑝_𝐵𝑁S
        
        #JO
        Sw=2*np.sum(group['easy']-group['close'])
        ome=np.sum(group['close'].rolling(4).apply(prodo))
        ome=(mu6/9)*240**3/237*mu32**(-4)*ome
        sqrt_ome = np.sqrt(ome)
        if sqrt_ome != 0:
            To_part1 = Bv / (sqrt_ome / 240)
            To_part2 = 1 - Rv / Sw
            To = To_part1 * To_part2
        else:
            To = None  # 设置一个合适的默认值

        #I_JO
        if To==None:
            i1=None
        elif To>1/phi:
            i1=1
        else:i1=0

        lc.append(code)
        
        #计算roll前的TSRJVP和JAR
        lTSRJVP.append(np.max(RVp-IV/2,0))
        lJAR.append(abs(ret) * i1 if i1 is not None else None)
    df=pd.DataFrame({'code':lc,'mid_JAR':lJAR,'mid_TSRJVP':lTSRJVP}).set_index('code')
    return df
def calc_retimp(trading_day,df):
    print('retimp')
    groups=df[['order_book_id','trade_time','close','volume']].groupby('order_book_id')
    lc=[]
    lr=[]
    for code, group in groups:
        group['ret']=group['close']/group['close'].shift(1)-1 #收益率
        vol=group['volume'].mean()+group['volume'].std() #放量标准
        group=group[group['volume']>vol]  #放量时段
        ret_imp=-group['ret'].std()  #取负
        lr.append(ret_imp)
        lc.append(code)
    df=pd.DataFrame({'code':lc,'ret_imp':lr}).set_index('code') 
    return df
def calc_TTV(trading_day,df,total_amount):
    day=int(trading_day.replace('-',''))
    print('TTV')
    df=df.set_index('trade_time')
    df.index=pd.to_datetime(df.index)
    df=df.between_time('14:31', '15:00')
    #total_amount=df.merge(total_amount,how='inner')
    groups=df[['order_book_id','total_turnover']].groupby('order_book_id')['total_turnover'].sum()/(-total_amount.loc[day])
    df=pd.DataFrame({'code':groups.index,'TTV':groups.values}).set_index('code') 
    return df
#计算收益率的日波动率分布因子
def calc_uid(trading_day,df):
    print('uid')
    groups=df[['order_book_id','trade_time','close','volume']].groupby('order_book_id')
    lc=[]
    lv=[]
    for code, group in groups:
        group['ret']=group['close']/group['close'].shift(1)-1 #收益率
        vol_daily=group['ret'].std()  #放量标准
        lv.append(vol_daily)
        lc.append(code)
    df=pd.DataFrame({'code':lc,'uid':lv}).set_index('code') 
    return df
#dw统计量相关因子
def calc_dw(trading_day,df): #str - -
    date=trading_day.replace('-','')
    print('dw')
    close = df.pivot(index = 'trade_time', columns='order_book_id', values='close')
    volume = df.pivot(index = 'trade_time', columns='order_book_id', values='volume')
    close.index=pd.to_datetime(close.index)
    volume.index=pd.to_datetime(volume.index)
    close.drop([x for x in close.columns if x[0]=='8'], axis=1,inplace=True)
    volume.drop([x for x in volume.columns if x[0]=='8'], axis=1,inplace=True)
    close=close.replace(0,np.nan)
    close=close.bfill(axis=0)
    close = close.pct_change().dropna()
    close = close.between_time('9:41','14:50')
    volume = volume.between_time('9:41','14:50')
    if close.isna().any().any() or volume.isna().any().any():
        print(date+'kong!!!!!!!!!!')
    if np.isinf(close).any().any():
        print(date+'inf!!!!!!!!!!')
        
    # 计算一阶自相关残差
    dfr = close - close.apply(lambda x: sm.tsa.AutoReg(x,lags=1,missing='drop').fit().fittedvalues)
    dfv = volume - volume.apply(lambda x: sm.tsa.AutoReg(x, lags=1,missing='raise').fit().fittedvalues)
    # 计算dw统计量
    dfr = np.power((close - close.shift(1)),2).sum() / (dfr * dfr).sum()
    dfv = np.power((volume - volume.shift(1)),2).sum() / (dfv * dfv).sum() 
    out=dfr+dfv
    out = out.to_frame().reset_index()
    out.columns=['code', 'com_dw']
    out=out.set_index('code')
    return out
#Q统计量相关因子
def calc_LBQ(trading_day,df): #str - -
    date=trading_day.replace('-','')
    close = df.pivot(index = 'trade_time', columns='order_book_id', values='close')
    volume = df.pivot(index = 'trade_time', columns='order_book_id', values='volume')
    close.index=pd.to_datetime(close.index)
    volume.index=pd.to_datetime(volume.index)
    close.drop([x for x in close.columns if x[0]=='8'], axis=1,inplace=True)
    volume.drop([x for x in volume.columns if x[0]=='8'], axis=1,inplace=True)
    #计算收益率
    close = close.pct_change().dropna()
    # 剔除开盘收盘半小时的数据
    close = close.between_time('9:41','14:50')
    volume = volume.between_time('9:41','14:50')
    dfr = close.apply(lambda x: acorr_ljungbox(x)['lb_stat'].std())
    dfv = volume.apply(lambda x: acorr_ljungbox(x)['lb_stat'].std())

    out=dfr+dfv
    out = out.to_frame().reset_index()
    out.columns=['code', 'com_LBQ']
    out=out.set_index('code')
    return out
#计算高波收益率均值因子
def calc_high(trading_day,df): #str - -
    df=df.set_index('trade_time')
    df.index=pd.to_datetime(df.index)
    df.drop([x for x in df.columns if x[0]=='8'], axis=1,inplace=True)
    df=df.between_time('9:40','14:50')[['order_book_id','close']]
    groups=df.groupby('order_book_id')
    lr=[]
    lc=[]
    for i,group in groups:
        g5=group.resample('5T').asfreq()
        g5['ret']=(g5['close']-g5['close'].shift(1))/g5['close'].shift(1)
        ggs=g5['ret'].rolling(6).std()
        ggs=ggs.dropna(how='all',axis=0)
        ggs=ggs[ggs.values>ggs.quantile(0.8)]
        ret=ggs.mean()
        lr.append(ret)
        lc.append(i[:6])

    dfr=pd.DataFrame({'code':lc,'highret':lr}).set_index('code')
    return dfr

# #进行日度计算
# def daily_factor_total(start_date, end_date):
#     trade_dates = pd.read_csv(path_trade_date)
#     trade_dates.Date = pd.to_datetime(trade_dates.Date)
#     trade_date = trade_dates[(trade_dates.Date >= start_date) & (trade_dates.Date <= end_date)]
#     trade_date.Date = trade_date.Date.astype('str')
#     for trading_day in trade_date.Date:
#         date=trading_day.replace('-','')
#         if os.path.exists(save_daily+'/'+date+'.fea'):
#             continue
#         #读当日分钟数据
#         print(trading_day)
#         df=pd.read_parquet(readpath+trading_day+'.parquet')
#         df=df[~df['order_book_id'].str.startswith('8')]
#         df['order_book_id']=df['order_book_id'].str[:6]
#         #因子计算
#         retimp=calc_retimp(trading_day,df)
#         TTV=calc_TTV(trading_day,df,total_amount)
#         com_dw=calc_dw(trading_day,df)
#         uid=calc_uid(trading_day,df)
#         com_LBQ=calc_LBQ(trading_day,df)
#         highret=calc_high(trading_day,df)
#         jump=calc_ITIV(trading_day,df)
#         #写文件
#         output=pd.concat([retimp,TTV,highret,uid],axis=1)
#         output.to_feather(save_daily+'/'+date+'.fea')
#         midput=pd.concat([com_dw,com_LBQ,jump],axis=1)
#         midput.to_feather(save_mid+'/'+date+'.fea')
 
#日度因子合成为历史因子
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
#roll计算生成entire因子
def calc_roll_to_entire():
    df=pd.read_csv('/data/disk4/output_stocks/jmchen/factors/minutes/mid_entire_files/com_dw.csv')
    df=df.set_index('date')
    df=df.rolling(15).mean()
    df.to_csv(save_factors+'/com_dw.csv')
    df=pd.read_csv('/data/disk4/output_stocks/jmchen/factors/minutes/mid_entire_files/com_LBQ.csv')
    df=df.set_index('date')
    df=df.rolling(15).mean()
    df.to_csv(save_factors+'/com_LBQ.csv')
    df=pd.read_csv('/data/disk4/output_stocks/jmchen/factors/minutes/mid_entire_files/mid_TSRJVP.csv')
    df=df.set_index('date')
    df=df.rolling(20).mean()
    df.to_csv(save_factors+'/TSRJVP.csv')
    df=pd.read_csv('/data/disk4/output_stocks/jmchen/factors/minutes/mid_entire_files/mid_JAR.csv')
    df=df.set_index('date')
    df=df.rolling(20).sum()
    df=np.exp(df)-1
    df.to_csv(save_factors+'/JAR.csv')
#entire因子转化为daily（主要用于roll后因子转daily）
def entire_to_daily(trading_day,factors):
    date=int(trading_day.replace('-',''))
    if os.path.exists(save_roll_daily+'/'+str(date)+'.fea'):
        return
    files=os.listdir('/data/disk4/output_stocks/jmchen/factors/minutes/entire_files')
    dfs=[]
    for file in files:
        if any(file.split('.')[0] in factor for factor in factors): #只生成需要roll的daily
            df=pd.read_csv(save_factors+'/'+file).set_index('date')
            dd=df.loc[date,:]
            dd=dd.rename(file.split('.')[0])
            dfs.append(dd)
    df=pd.concat(dfs,axis=1).rename_axis('code')
    df.to_feather(save_roll_daily+'/'+str(date)+'.fea')

#进行日度计算
def daily_factor(trading_day):
    date=trading_day.replace('-','')
    if os.path.exists(save_daily+'/'+date+'.fea'):
        return 
    #读当日分钟数据
    print(trading_day)
    df=pd.read_parquet(readpath+trading_day+'.parquet')
    df=df[~df['order_book_id'].str.startswith('8')]
    df['order_book_id']=df['order_book_id'].str[:6]
    #因子计算
    retimp=calc_retimp(trading_day,df)
    TTV=calc_TTV(trading_day,df,total_amount)
    com_dw=calc_dw(trading_day,df)
    uid=calc_uid(trading_day,df)
    com_LBQ=calc_LBQ(trading_day,df)
    highret=calc_high(trading_day,df)
    jump=calc_ITIV(trading_day,df)
    #写文件
    output=pd.concat([retimp,TTV,highret,uid],axis=1)
    output.to_feather(save_daily+'/'+date+'.fea')
    midput=pd.concat([com_dw,com_LBQ,jump],axis=1)
    midput.to_feather(save_mid+'/'+date+'.fea')
    return

# ### 全部刷新部分
# start_date=datetime.strptime('20240131','%Y%m%d')
# end_date=datetime.strptime('20240228','%Y%m%d')
# trade_dates = pd.read_csv(path_trade_date)
# trade_dates.Date = pd.to_datetime(trade_dates.Date)
# trade_date = trade_dates[(trade_dates.Date >= start_date) & (trade_dates.Date <= end_date)]
# trade_date.Date = trade_date.Date.astype('str')
# num_processes = 13
# # 创建进程池
# args=[trading_day for trading_day in trade_date.Date]
# pool = multiprocessing.Pool(processes=num_processes)
# results=pool.map(daily_factor,args)

#每日更新部分
trade_dates = pd.read_csv(path_trade_date)
trade_dates['Date']=pd.to_datetime(trade_dates['Date'])
trade_dates['Date'] = trade_dates['Date'].dt.date     
today=date.today()
today=today-timedelta(1)  #每天早上5:30更新，由前一天数据算因子值
print(today)
flag=0
for i in trade_dates.index:
    if trade_dates.loc[i,'Date']==today:
        print('交易日更新分钟因子')
        flag=1
if flag==0:
    print('非交易日因子无更新')
    sys.exit()
trading_day=datetime.strftime(today,'%Y-%m-%d')
daily_factor(trading_day)

rolling_dic={'ret_imp':1,'TTV':1,'com_dw':15,'com_LBQ':15,'highret':1,'uid':1,'mid_JAR':20,'mid_TSRJVP':20}
factors= [k for k, v in rolling_dic.items() if v == 1] #直接拼不用roll的
combine(factors=factors)
factors= [k for k, v in rolling_dic.items() if v > 1]  #要roll的拼到mid_entire
combine(factors=factors,daily_path=save_mid,factpath=save_mid_entire)

calc_roll_to_entire()
# start_date=datetime.strptime('20200102','%Y%m%d')
# end_date=datetime.strptime('20240228','%Y%m%d')
# trade_dates = pd.read_csv(path_trade_date)
# trade_dates.Date = pd.to_datetime(trade_dates.Date)
# trade_date = trade_dates[(trade_dates.Date >= start_date) & (trade_dates.Date <= end_date)]
# trade_date.Date = trade_date.Date.astype('str')
# for trading_day in trade_date.Date:
#     print(trading_day)
entire_to_daily(trading_day,factors=[k for k, v in rolling_dic.items() if v > 1])

receivers.remove('657288788@qq.com')
receivers.remove('ftluo@ksquant.com.cn')
send_mail( msg=trading_day+'分钟因子每日更新完成')