import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import OrderedDict
import sys
sys.path.append("/home/jmchen/files")
from demo_email import send_mail
from demo_email import receivers
from datetime import datetime,timedelta

rootpath='/data/disk3/DataBase_stocks/riceQuantData/minute_price/CS/'
savepath='/data/disk4/DataBase_stocks/jmchenData/twap/twap_daily_interval/'

def calc_twap(df):
    df_high = df['high']
    df_low = df['low']
    twap = (df_high + df_low) / 2
    twap_sum = twap.cumsum()
    df['twap_sum'] = twap_sum / (df.index.to_series().rank())
    return df
#groupby后的df
def cut_df(df,interval=30):
    devided_df = [group for _, group in df.groupby(df.index // interval)]
    return devided_df

# #刷历史数据
# for dayname in os.listdir(rootpath):
#     print(dayname)
#     day=dayname[:10]
#     print(day)
#     if os.path.exists(savepath+day+'.csv'):
#         continue
#     df=pd.read_parquet(rootpath+dayname)
#     code = df['order_book_id'].tolist()
#     unique_code = list(OrderedDict.fromkeys(code))

#     grouped_df = df.groupby('order_book_id')
#     results=[]
#     for i, group in tqdm(grouped_df, desc="处理中"): #每支股票按时间分组
#         group.reset_index(drop=True,inplace=True)
#         devided=cut_df(group)
#         twaps=[]
#         for devide in devided:
#             df=calc_twap(devide)
#             twaps.append(df.iloc[-1]['twap_sum']) #每个时间段的数据列表
#         results.append(twaps)

#     dfout= pd.DataFrame(results, columns=[
#         '9:31-10:00', '10:01-10:30', '10:31-11:00','11:01-11:30','13:01-13:30','13:31-14:00','14:01-14:30','14:31-15:00'])
#     dfout.insert(0,column='code',value=unique_code)
#     dfout.to_csv(savepath+day+'.csv')




day=datetime.now()
dayname=day.strftime('%Y-%m-%d')
df=pd.read_parquet(rootpath+dayname+'.parquet')

code = df['order_book_id'].tolist()
unique_code = list(OrderedDict.fromkeys(code))

grouped_df = df.groupby('order_book_id')
results=[]
for i, group in tqdm(grouped_df, desc="处理中"): #每支股票按时间分组
    group.reset_index(drop=True,inplace=True)
    devided=cut_df(group)
    twaps=[]
    for devide in devided:  
        df=calc_twap(devide)
        twaps.append(df.iloc[-1]['twap_sum']) #每个时间段的数据列表
    results.append(twaps)

dfout= pd.DataFrame(results, columns=[
    '9:31-10:00', '10:01-10:30', '10:31-11:00','11:01-11:30','13:01-13:30','13:31-14:00','14:01-14:30','14:31-15:00'])
dfout.insert(0,column='code',value=unique_code)
dfout.to_csv(savepath+dayname+'.csv',encoding="utf_8_sig")
receivers.remove('657288788@qq.com')
receivers.remove('ftluo@ksquant.com.cn')
send_mail( msg=dayname+'twap分时段更新完成')

