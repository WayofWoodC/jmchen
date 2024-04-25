#%%
import sys  
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker # 导入设置坐标轴的模块
import os,time,gc,psutil,sys ,json,pyarrow
from concurrent.futures import ThreadPoolExecutor
pol=ThreadPoolExecutor(max_workers=3)
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import pandas as pd
import itertools
from tqdm.auto import tqdm,trange #
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

#因子存储根目录
save_dir='/data/disk4/output_stocks/jmchen/factors/shot/shanghai'
#日度因子更新存储路径
save_daily=save_dir+'/Factor_save_daily'
#单因子的历史数据
save_factors=save_dir+'/Factors_all'
#读取数据目录
readpath=r'/data/disk3/DataBase_stocks/marketData_kuaizhao'

#本因子计算用到的中间路径变量
save_args={'mid_path':os.path.join(save_dir,'Factor_middle'),  
        'save_path':os.path.join(save_dir,'Factor_save_preFac'),
        'daily_path':os.path.join(save_dir,'Factor_save_daily'),}
#
#
#
#%%#因子计算方式未降维
class factors_initiate():
    def __init__(self,periods=20): #分钟长度roll
        self.factors={}      
        def preV(df):  #每笔成交均量
            Pvft=df['volume'].astype(float)/df['num_trades']
            Pvft=Pvft.ffill().rolling(periods, min_periods=1).mean().replace(np.inf,0).fillna(0)
            return Pvft.to_frame()
        self.factors['preV']={'cols':['volume','num_trades'],'func':preV}

        def LQS(df):  #成交概率反比？
            df=df.apply(np.log)  #取对数
            LQSft=(df['ask0']-df['bid0'])/(df['ask_volume0']-df['bid_volume0'])
            LQSft=LQSft.ffill().rolling(periods, min_periods=1).mean().replace(np.inf,0).fillna(0)
            return LQSft.to_frame()
        self.factors['LQS']={'cols':['bid0','bid_volume0','ask0','ask_volume0'],'func':LQS}

        level1=0;level10=9
        def SOIR(df): #（Size of Imbalance Ratio）
            fn=pd.Series(index=df.index,dtype=float).fillna(0)
            for i in range(level1,level10+1):
                fn+=((df['bid_volume%s'%i]-df['ask_volume%s'%i])/(df['bid_volume%s'%i]+df['ask_volume%s'%i])).fillna(0)*(level10-i)  #加权
            fn/=(level1+level10)*(level10+1)/2  #归一化
            fn=fn.ffill().replace(np.inf,0).fillna(0)
            mean =fn.shift(1).rolling(periods-1, min_periods=2).mean()
            std = (fn.shift(1)*1000000).rolling(periods-1, min_periods=2).std()/ 1000000
            fn=(fn - mean) / std
            fn = fn.where(std >= 0.0000001,).ffill().replace(np.inf,0).fillna(0)
            return fn.to_frame()
        self.factors['SOIR']={'cols': [vv+ str(x) for vv in ('bid_volume','ask_volume') for x in range(level1,level10+1)],'func':SOIR}

        def NBQS(df): ###难以复刻  #负相关
            df= df.diff()       #差分，变化量
            avgp=df['total_turnover']/df['total_volume']  #成交量变化引起的价格变化
            WNB= (df['bid_volume0' ]-df['ask_volume0' ]  ) / (df['bid_volume0' ].abs()+df['ask_volume0' ].abs()  )  #买1卖1成交量变化的偏离程度
            fn=(WNB*avgp).rolling(periods, min_periods=1).sum()/(avgp.rolling(periods, min_periods=1).sum())  #买卖1成交量变化与均价变化的关系
            return fn.to_frame()
        self.factors['NBQS']={'cols': 'bid_volume0,ask_volume0,total_turnover,total_volume'.split(','),'func':NBQS}
        
        def WNBQSdol(df): ###难以复刻
            
            bidchgyy=(-df['bid_volume0' ]).where(round( df['bid_volume0' ].diff(),2)<0,df['bid_volume0' ].diff())
            bidchg=df['bid_volume0' ].where(round( df['bid_volume0' ].diff(),2)>0,bidchgyy)
            of1=(df['ask_volume0'].diff()>0).where(df['ask_volume0']!=0,(df['ask_volume0'].shift(1)>0))
            of313= (df['ask_volume0'].diff()<0).where(df['ask_volume0'].shift(1)<0,False)
            of31=(df['ask_volume0']>0).where( df['ask_volume0'].shift(1)==0 , of313 )
            of3=df['ask_volume0'].where( of31, df['ask_volume0'].diff() )
            offerchg=df['ask_volume0'].shift(1).where(of1,of3)
    
            avgp=df['total_turnover'].diff()/df['total_volume'].diff()
            factorValue=(bidchg-offerchg)*avgp/(bidchg.abs()+offerchg.abs())
            mavgprice=avgp.rolling(periods, min_periods=1).mean()
            mfactorValue=(factorValue.rolling(periods, min_periods=1).mean()/mavgprice).replace(np.inf,0).fillna(0)
            del bidchgyy,bidchg,of1,of313,of31,of3,offerchg,avgp,factorValue,mavgprice
            return mfactorValue.to_frame()
        self.factors['NBQdol']={'cols': 'bid_volume0,ask_volume0,total_turnover,total_volume'.split(','),'func':WNBQSdol}
        
        def LTD(df):
            df['bid0d']=df['bid0'].shift(1)
            df['bid9d']=df['bid9'].shift(1)
            hb=df[['bid0','bid0d']].max(axis=1) #最近6秒的最高买1价     #逻辑有问题
            lb=df[['bid9','bid9d']].max(axis=1) #最近6秒的最高买9价   
            bids=df[['bid%s'%i for i in range(level1,level10+1) ]]
            df[['bid%s'%i for i in range(level1,level10+1) ]]=bids.where( bids.ge(lb, axis='index') & bids.le(hb, axis='index')).fillna(0)  #现在只去掉了6秒中的最低报价
            fn=pd.Series(index=df.index,dtype=float).fillna(0)
            for i in range(level1,level10+1):
                fn+=df['bid%s'%i]*df['bid_volume%s'%i]  #去掉极值的意愿成交额
            fn=fn.diff().rolling(periods, min_periods=1).mean().ffill().replace(np.inf,0).fillna(0)
            return fn.to_frame()
        self.factors['LTD']={'cols': [vv+ str(x) for vv in ('bid_volume','bid') for x in range(level1,level10+1)] ,'func':LTD}

        def LTDnew(df):
            df['bid0d']=df['bid0'].shift(1)
            df['bid9d']=df['bid9'].shift(1)
            hb=df[['bid0','bid0d']].min(axis=1) #最近6秒的最低买1价     
            lb=df[['bid9','bid9d']].max(axis=1) #最近6秒的最高买9价   
            bids=df[['bid%s'%i for i in range(level1,level10+1) ]]
            df[['bid%s'%i for i in range(level1,level10+1) ]]=bids.where( bids.ge(lb, axis='index') & bids.le(hb, axis='index')).fillna(0)  #现在只去掉了6秒中的最低报价
            fn=pd.Series(index=df.index,dtype=float).fillna(0)
            for i in range(level1,level10+1):
                fn+=df['bid%s'%i]*df['bid_volume%s'%i]  #去掉最值的中坚力量买单意愿成交额
            fn=fn.diff().rolling(periods, min_periods=1).mean().ffill().replace(np.inf,0).fillna(0)
            return fn.to_frame()
        self.factors['LTD']={'cols': [vv+ str(x) for vv in ('bid_volume','bid') for x in range(level1,level10+1)] ,'func':LTD}
        
        def IFP(df):
            bd=df[['bid'+ str(x) for x in range(level1,level10+1)] ].values
            bdv=df[['bid_volume'+ str(x) for x in range(level1,level10+1)] ].values
            ak=df[['ask'+ str(x) for x in range(level1,level10+1)] ].values
            akv=df[['ask_volume'+ str(x) for x in range(level1,level10+1)] ].values
            kk=(bd*bdv+ak*akv).sum(axis=1)/(bdv+akv).sum(axis=1)  #前10档买卖vwap
            def regression(Y): #回归计算
                IFP_period=60
                X=np.c_[np.ones(IFP_period),np.array(range(IFP_period)),]
                rr=(np.linalg.inv(((X.T) @ X)) @ ((X.T) @  Y.values))  #@是矩阵乘法运算
                return rr[1]  #返回斜率
            #kk=pd.Series(kk,index=df.index).rolling(IFP_period).apply(regression).rolling(periods, min_periods=1).mean().ffill().replace(np.inf,0).fillna(0)
            return kk.to_frame()
        self.factors['IFP']={'cols': [vv+ str(x) for vv in ('bid_volume','bid','ask_volume','ask') for x in range(level1,level10+1)] ,'func':IFP}

        def Bna(df):
            fn=(df['total_bid_volume']-df['total_ask_volume']).diff().ffill().rolling(periods, min_periods=1).mean().replace(np.inf,0).fillna(0)
            return fn.to_frame()
        self.factors['Bna']={'cols':['total_bid_volume','total_ask_volume'],'func':Bna}

def calc_initial_factors(name,fac_dic,df):
        T1=time.time()
        factor=fac_dic['Class'][name]
        temps=[]
        batch_size=200
        codes=df.security_id.drop_duplicates().index
        for i in range((len(codes)-1)//batch_size+1):
            start=codes[i*batch_size]
            if  (i+1)*batch_size<len(codes):
                        end=codes[(i+1)*batch_size-1]
            else:
                        end=df.index[-1]
            temp=df[['security_id']+factor['cols']].loc[start:end].groupby('security_id',   #拆块会算更快吗？？？
             group_keys=False)[factor['cols']].apply( factor['func'])
            temps.append(temp)
        initial_factors=pd.concat(temps)
        initial_factors.columns=['initial_value']
        initial_factors=pd.concat([df[['security_id','trade_time']],initial_factors],axis=1)

        print(name,round(time.time()-T1,2),end=',')
        #print(initial_factors)
        return initial_factors

def calc_final_factors(fac_arg,initial_factors):
    T1=time.time()
    method= getattr(pd.Series, fac_arg['Ded'])
    final_factors=initial_factors[['security_id','initial_value']].groupby('security_id')['initial_value'].apply(method)
    final_factors.name='因子值'
    print(fac_arg['Ded'],round(time.time()-T1,2),end=',')
    return final_factors.to_frame().reset_index()
#%% 数据def准备
def find_dates(readpath):  #返回日期型变量
    快照文件表=os.listdir(readpath)
    快照文件表.sort()
    快照文件z=pd.DataFrame([ os.path.splitext(filename) for filename in 快照文件表],columns=['日期','后缀'])
    行情数据位置=r'/data/disk1/data/DataBase_stock/dailyData/'
    行情文件表=os.listdir(行情数据位置)
    行情文件表.sort()
    行情文件z=pd.DataFrame([ os.path.splitext(filename) for filename in 行情文件表],columns=['日期','后缀'])
    有效行情日期=pd.to_datetime(行情文件z.日期)
    有无文件=有效行情日期.isin(快照文件z.日期)
    for idx in 有无文件.index[::-1]:
        if 有无文件[idx]:break
    有无文件=有无文件[:idx+1]
    tradedates=pd.Index(pd.to_datetime(有效行情日期[  有无文件[~有无文件].index[-1]+1 :  有无文件.index[-1]+1 ]))
    return tradedates
def read_df(date,readpath):
    try:
                df=pd.read_parquet(os.path.join(readpath,date.strftime('%Y-%m-%d')+'.parquet'),)
                return df
    except pyarrow.lib.ArrowInvalid as e:
                print(date,'文件损坏！')
                return None

if __name__ == '__main__':
    #
    tradedates=find_dates(readpath=readpath)
    fac_arg={}
    factors=factors_initiate().factors
    
    fac_dic={
            'Class':factors,
            }
    for i in  save_args.keys(): os.makedirs(save_args[i],exist_ok=True)
    fixed_factor=True                                  #是否指定因子和处理为以下组合
    methods=['mean','std','median','skew','kurtosis',] #单维
    #['preV', 'LQS', 'SOIR', 'NBQS', 'NBQdol', 'LTD', 'IFP', 'Bna']
    factor_groups=[
        ['NBQS','kurtosis'],#good                 负相关，21年市场原因，IC均值0.06，单调性好
        ['NBQS','std'],#ok分离度可以，top3差了点    正相关 21年单调性差点，因子整体分布较为正态 IC均值0.07，超额较好
        ['LTD','std',],#very good 最猛的一个       负相关，21年市场原因, IC>0.07,单调性好，1e8   逻辑问题
        ['LTD','skew'],#soso                      负相关，一般
        ['preV','kurtosis',],#very good           负相关，单调性好，IC0.05
        ['preV','skew',],#very good               差不多
        ['SOIR','std',],#ok                       正相关，IC0.05，单调性22年最好，10分组排top3   待去极大值
        ['SOIR','median'],#ok                     负相关，不行,IC低
        ['LQS','kurtosis',],#so so                负相关，一般
        ['LQS','std'],#soso                       正相关，一般
        ['NBQdol','std'],#very good               负相关，一般，分布
        ['NBQdol','kurtosis'],#good               负相关，不行，IC低
    ]
    #负相关因子降维时转正
    judge_groups=[
        ['NBQS','kurtosis'],#good                 负相关，21年市场原因，IC均值0.06，单调性好
        ['LTD','std',],#very good 最猛的一个       负相关，21年市场原因, IC>0.07,单调性好，1e8   逻辑问题
        ['LTD','skew'],#soso                      负相关，一般
        ['preV','kurtosis',],#very good           负相关，单调性好，IC0.05
        ['preV','skew',],#very good               差不多,负相关
        ['SOIR','median'],#ok                     负相关，不行,IC低
        ['LQS','kurtosis',],#so so                负相关，一般
        ['NBQdol','std'],#very good               负相关，一般，分布
        ['NBQdol','kurtosis'],#good               负相关，不行，IC低
    ]
    
    if not fixed_factor: 
        factor_groups=list(itertools.product(['preV', 'LQS', 'SOIR', 'NBQS','NBQdol',  'LTD',  'Bna'], methods))
    factor_groups=pd.DataFrame(factor_groups,columns=['name','method']).sort_values(by=['name'], ignore_index=True)

    
    try: 
        sequence=int(sys.argv[1])        
    except :
        sequence=-1 
    #
    for date in tradedates[::sequence]:
        date_=date.strftime('%Y-%m-%d')
        Date=date.strftime('%Y%m%d')
        save_file=os.path.join(save_args['daily_path'],Date+'.fea')
        if (not os.path.exists(save_file))  and  fixed_factor:     #不重复写因子文件
            save_df=pd.DataFrame()
        else: save_df=None
        print(date_,end=' ')
        df=None
        #df在循环外面读
        i_name=os.path.join(save_args['mid_path'],factor_groups.loc[0,'name'],date_+'.fea')
        t1=time.time()

        if not os.path.exists(i_name): 
            if df is None: 
                print('开始读取快照',end=',')
                T1=time.time()
                df=read_df(date,readpath=readpath)##没就读
                print('快照:',round(time.time()-T1,2),end=',')
    
        for name,m in factor_groups.groupby('name'):
            fac_arg['Fac']=name
            initial_factors=None
            ###未降维
            os.makedirs(os.path.join(save_args['mid_path'],name),exist_ok=True)
            i_seperate_name=os.path.join(save_args['mid_path'],name,date_+'.fea')
            if not os.path.exists(i_seperate_name): 
                if df is None: continue###坏了就跳
                initial_factors=calc_initial_factors(name,fac_dic,df)
                initial_factors.to_feather(i_seperate_name)
            ###降维
            for way in m['method'].tolist():   
                fac_arg['Ded']=way
                fac_dic['number']='_'.join([ str(x)[:3]+str(fac_arg[x]) for x in fac_arg.keys()])
                fac_path=os.path.join(save_args['save_path'],fac_dic['number'])
                os.makedirs(fac_path,exist_ok=True)
                seperate_name=os.path.join(fac_path,date_+'.fea')
                #判断是否需要负号用
                judge_group=[]
                judge_group.append(fac_arg['Fac'])
                judge_group.append(fac_arg['Ded'])       
                if not os.path.exists(seperate_name): 
                    if initial_factors is None:
                        print('开始读取中间',end=',')
                        T1=time.time()
                        initial_factors=pd.read_feather(i_seperate_name)  
                        print('中间:',round(time.time()-T1,2),end=',')
                    #降维计算日度因子值
                    daily_factor=calc_final_factors(fac_arg,initial_factors)
                    if judge_group in judge_groups: #判断因子值是否需要负号
                        daily_factor['因子值']=-daily_factor['因子值']
                    daily_factor.to_feather(seperate_name)
                elif save_df is not None:
                    daily_factor=pd.read_feather(seperate_name)
                
                if save_df is not None:
                    daily_factor.rename(columns={'因子值':fac_dic['number'],'initial_value':fac_dic['number']},inplace=True)
                    save_df=pd.concat([save_df,daily_factor.set_index('security_id')],axis=1)
                
        if save_df is not None:
                save_df.index=save_df.index.str[:6]
                save_df.reset_index().to_feather(save_file)
        del initial_factors,df;gc.collect()
        #sys.exit()   
        print(Date,'\n')
        print(time.time()-t1)
        
###
# %%
