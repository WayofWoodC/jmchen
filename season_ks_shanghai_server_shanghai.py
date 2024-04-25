#
import pandas as pd
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import seaborn as sns
import datetime as dt
from datetime import datetime,timedelta#,date
import os
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

#basepath = 'X:/'
basepath ='/data/'

rootpath = basepath + 'DataBase_stock/joinQuantData/risk_factor/'
savepath = basepath + 'output_stocks/季度报/results_forMarket/'
os.makedirs(savepath, exist_ok = True)

# 板块表现
shares_path = basepath + 'DataBase_stock/riceQuantData/shares/' # 总股本
close_path = basepath + 'DataBase_stock/riceQuantData/eod_price/CS/' # close价格
industry_path = basepath + 'DataBase_stock/riceQuantData/instrument_industry/citics_2019/' # 行业分类

this_month =datetime.now().date().month
### 可手动修改，默认一个季度结束后在新的月执行
#season=1  #第几季度
season=this_month//3 

###生成表格的列名和文件名，不需要手动修改
first_month=str(1+(season-1)*3)
second_month=str(2+(season-1)*3)
third_month=str(3+(season-1)*3)
indus_name='24年第'+str(season)+'季度.csv'
barra_name='24年barra'+first_month+'-'+third_month+'.csv'

f = sorted(os.listdir(basepath + 'DataBase_stock/riceQuantData/eod_price/CS/'))
#f.sort()
print(pd.Series(f))
month = [i[:7] for i in f][::-1]
unimonth = pd.unique(month).tolist()



#month
#改下标，取到月末日期
# new_index=-1
# old_first_index = -22 
# old_second_index=-37  
# old_last_index=-59

today = str(datetime.now().date())
print('today:', today)
today_month = today[:7]
pre1_month = unimonth[unimonth.index(today_month) + 1]
pre2_month = unimonth[unimonth.index(today_month) + 2]
pre3_month = unimonth[unimonth.index(today_month) + 3]
pre4_month = unimonth[unimonth.index(today_month) + 4]
f[- (month.index(pre1_month)+1) ]
f[- (month.index(pre2_month)+1) ]
f[- (month.index(pre3_month)+1) ]
f[- (month.index(pre4_month)+1) ]

new_csv = f[- (month.index(pre1_month)+1) ]#f[new_index]         #  上月末（季度最后一个月末） 2024-03-29.csv
old_first_csv = f[- (month.index(pre2_month)+1) ]#f[old_first_index]    #  上上月末 2024-02-29.csv
old_second_csv = f[- (month.index(pre3_month)+1) ]#f[old_second_index]   #季度第一个月末 2024-01-31.csv
old_last_csv = f[- (month.index(pre4_month)+1) ]#f[old_last_index]     #上上个季度末 2023-12-29.csv
print('上月末（季度最后一个月末）:',new_csv)
print('上上月末:', old_first_csv)
print('季度第一个月末:', old_second_csv)
print('上上个季度末:', old_last_csv)  

new_index = f.index(new_csv) #-1
old_first_index = f.index(old_first_csv)#-22 
old_second_index = f.index(old_second_csv)#-37  
old_last_index = f.index(old_last_csv)#-59


# 检查并处理重复索引
def remove_duplicate_index(df):
    duplicated_indices = df.index.duplicated()
    if any(duplicated_indices):
        print("存在重复的索引值，将保留第一个出现的索引值。")
        df = df[~duplicated_indices]
    return df
#

#%%计算板块振幅
file = f[old_last_index]
print('old_last_index:', file)

close = pd.read_csv(close_path + file, index_col = 0)
shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
indus = pd.read_csv(industry_path + file, index_col = 0)
#删除重复索引情况
indus=remove_duplicate_index(indus)
all = pd.concat([indus.first_industry_name, shares.total, close.close], axis = 1).dropna()

all['value'] = all.close * all.total
final = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
big = pd.Series(0, index=final.index)
small=pd.Series(np.Inf,index=final.index)
for file in f[old_last_index:new_index]:  #季度整体情况
    close = pd.read_csv(close_path + file, index_col = 0)
    shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
    indus = pd.read_csv(industry_path + file, index_col = 0)
    indus=remove_duplicate_index(indus)
    all = pd.concat([indus.first_industry_name, shares.total, close.close.rename('close')], axis = 1).dropna()
    all['value'] = all.close * all.total
    temp = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
    big=pd.Series(np.maximum(big,temp))
    small=pd.Series(np.minimum(small,temp))
finalall=(big-small)/final

file = f[old_first_index]
print('old_first_index:', file)

close = pd.read_csv(close_path + file, index_col = 0)
shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
indus = pd.read_csv(industry_path + file, index_col = 0)
indus=remove_duplicate_index(indus)
all = pd.concat([indus.first_industry_name, shares.total, close.close], axis = 1).dropna()
all['value'] = all.close * all.total
final = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
big = pd.Series(0, index=final.index)
small=pd.Series(np.Inf,index=final.index)
for file in f[old_first_index:new_index]: #最近一个月情况
    close = pd.read_csv(close_path + file, index_col = 0)
    shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
    indus = pd.read_csv(industry_path + file, index_col = 0)
    indus=remove_duplicate_index(indus)
    all = pd.concat([indus.first_industry_name, shares.total, close.close.rename('close')], axis = 1).dropna()
    all['value'] = all.close * all.total
    temp = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
    big=pd.Series(np.maximum(big,temp))
    small=pd.Series(np.minimum(small,temp))
final3=(big-small)/final

file=f[old_second_index]
print(file)
close = pd.read_csv(close_path + file, index_col = 0)
shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
indus = pd.read_csv(industry_path + file, index_col = 0)
indus=remove_duplicate_index(indus)
all = pd.concat([indus.first_industry_name, shares.total, close.close], axis = 1).dropna()
all['value'] = all.close * all.total
final = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
big = pd.Series(0, index=final.index)
small=pd.Series(np.Inf,index=final.index)
for file in f[old_second_index:old_first_index]:
    close = pd.read_csv(close_path + file, index_col = 0)
    shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
    indus = pd.read_csv(industry_path + file, index_col = 0)
    indus=remove_duplicate_index(indus)
    all = pd.concat([indus.first_industry_name, shares.total, close.close.rename('close')], axis = 1).dropna()
    all['value'] = all.close * all.total
    temp = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
    big=pd.Series(np.maximum(big,temp))
    small=pd.Series(np.minimum(small,temp))
final2=(big-small)/final

file=f[old_last_index]
print(file)
close = pd.read_csv(close_path + file, index_col = 0)
shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
indus = pd.read_csv(industry_path + file, index_col = 0)
indus=remove_duplicate_index(indus)
all = pd.concat([indus.first_industry_name, shares.total, close.close], axis = 1).dropna()
all['value'] = all.close * all.total
final = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
big = pd.Series(0, index=final.index)
small=pd.Series(np.Inf,index=final.index)
for file in f[old_last_index:old_second_index]:
    close = pd.read_csv(close_path + file, index_col = 0)
    shares = pd.read_csv(shares_path + file, index_col = 0).set_index("order_book_id")
    indus = pd.read_csv(industry_path + file, index_col = 0)
    indus=remove_duplicate_index(indus)
    all = pd.concat([indus.first_industry_name, shares.total, close.close.rename('close')], axis = 1).dropna()
    all['value'] = all.close * all.total
    temp = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.close)/ x.value.sum())
    big=pd.Series(np.maximum(big,temp))
    small=pd.Series(np.minimum(small,temp))
final1=(big-small)/final


#%%计算板块涨跌
files = sorted(os.listdir(basepath + 'DataBase_stock/winddb3/AINDEXEODPRICES_daily/'))
#files.sort()
# 板块表现(权重*收盘价变化率)
#shares_path = "/data/disk3/DataBase_stocks/riceQuantData/shares/" # 总股本
#close_path = "/data/disk3/DataBase_stocks/riceQuantData/eod_price/CS/" # close价格
#industry_path = "/data/disk3/DataBase_stocks/riceQuantData/instrument_industry/citics_2019/" # 行业分类


old_close = pd.read_csv(close_path + old_first_csv, index_col = 0)
new_shares = pd.read_csv(shares_path + new_csv, index_col = 0).set_index("order_book_id")
new_close = pd.read_csv(close_path + new_csv, index_col = 0)
new_indus = pd.read_csv(industry_path + new_csv, index_col = 0)
all = pd.concat([new_indus.first_industry_name, new_shares.total, old_close.close.rename('old_close'), new_close.close.rename('new_close')], axis = 1).dropna()
all['value'] = all.new_close * all.total
all['pct'] = (all.new_close - all.old_close) / all.old_close
first = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.pct)/ x.value.sum())

old_close = pd.read_csv(close_path + old_second_csv, index_col = 0)
new_shares = pd.read_csv(shares_path + old_first_csv, index_col = 0).set_index("order_book_id")
new_close = pd.read_csv(close_path + old_first_csv, index_col = 0)
new_indus = pd.read_csv(industry_path + old_first_csv, index_col = 0)
all = pd.concat([new_indus.first_industry_name, new_shares.total, old_close.close.rename('old_close'), new_close.close.rename('new_close')], axis = 1).dropna()
all['value'] = all.new_close * all.total
all['pct'] = (all.new_close - all.old_close) / all.old_close
second = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.pct)/ x.value.sum())

old_close = pd.read_csv(close_path + old_last_csv, index_col = 0)
new_shares = pd.read_csv(shares_path + old_second_csv, index_col = 0).set_index("order_book_id")
new_close = pd.read_csv(close_path + old_second_csv, index_col = 0)
new_indus = pd.read_csv(industry_path + old_second_csv, index_col = 0)
all = pd.concat([new_indus.first_industry_name, new_shares.total, old_close.close.rename('old_close'), new_close.close.rename('new_close')], axis = 1).dropna()
all['value'] = all.new_close * all.total
all['pct'] = (all.new_close - all.old_close) / all.old_close
third= all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.pct)/ x.value.sum())

old_close = pd.read_csv(close_path + old_last_csv, index_col = 0)
new_shares = pd.read_csv(shares_path + new_csv, index_col = 0).set_index("order_book_id")
new_close = pd.read_csv(close_path + new_csv, index_col = 0)
new_indus = pd.read_csv(industry_path + new_csv, index_col = 0)
all = pd.concat([new_indus.first_industry_name, new_shares.total, old_close.close.rename('old_close'), new_close.close.rename('new_close')], axis = 1).dropna()
all['value'] = all.new_close * all.total
all['pct'] = (all.new_close - all.old_close) / all.old_close
season = all.groupby('first_industry_name').apply(lambda x: sum(x.value * x.pct)/ x.value.sum())

###！！！此处修改输出板块数据的月份数字，以及表格文件名
if True:
    res = pd.concat([third.rename(first_month+'月涨跌幅'),final1.rename(first_month+'月振幅'),second.rename(second_month+'月涨跌幅'),
                     final2.rename(second_month+'月振幅'),first.rename(third_month+'月涨跌幅'),final3.rename(third_month+'月振幅'),
                     season.rename('季度涨跌幅'),finalall.rename('季度振幅')], axis = 1)
    res = res.sort_values('季度涨跌幅', ascending = False).reset_index().rename(columns = {'first_industry_name': '板块（中信行业一级分类）'})
    res.to_csv(savepath+indus_name,encoding="utf_8_sig")

print('季度涨跌 finished')


#%%barra收益计算
# 风格月度收益
sty_path = basepath + 'DataBase_stock/joinQuantData/risk_factor/'

files = sorted(os.listdir(sty_path))

new_index = files.index(new_csv) #-1
old_first_index = files.index(old_first_csv)#-22 
old_second_index = files.index(old_second_csv)#-37  
old_last_index = files.index(old_last_csv)#-59


selected_dict = {"average_share_turnover_annual":   "年度平均月换手率",
"average_share_turnover_quarterly":	"季度平均平均月换手率",
"beta":	"BETA",
"book_leverage":	"账面杠杆",
"book_to_price_ratio":	"市净率因子",
"cash_earnings_to_price_ratio":	"现金流量市值比",
"earnings_to_price_ratio":	"利润市值比",
"earnings_yield":	"盈利预期因子",
"growth":	"成长因子",
"historical_sigma":	"残差历史波动率",
"leverage":	"杠杆因子",
"liquidity": "流动性因子",
"momentum":	"动量因子",
"non_linear_size":	"非线性市值因子",
"residual_volatility":	"残差波动因子",
"size":	"市值因子"}
open_pct = pd.read_csv(basepath + 'DataBase_stock/AllSample/open.csv', index_col = 0).pct_change().shift(-1).dropna(how = 'all', axis = 1).dropna(how = 'all', axis = 0)
open_pct.columns = [i.split('.')[0] for i in open_pct.columns]
res = pd.DataFrame()

print(pd.Series(files[old_first_index:new_index]))

#第3个月
for file in tqdm(files[old_first_index:new_index]):  #每月1号晚上九点更新，否则若第二天则：-1
    print(file)
    sty = pd.read_csv(sty_path + file)
    
    date = int(file.replace('-', '').split('.')[0])
    
    sty = sty.drop(columns = 'trade_date').set_index('security_id')
    sty.index = [i.split('.')[0] for i in sty.index]
    
    ret = open_pct.loc[date]  ##########

    sty = sty.apply(lambda x: (x - x.mean()) / x.std())
    sty = sty.apply(lambda x: x / x.abs().sum())
    daily_res = sty.apply(lambda x: (x * ret).sum(), axis = 0).rename(date).to_frame()
    res = pd.concat([res, daily_res], axis = 1)
#
plt_res = res.T
plt_res = plt_res[list(selected_dict.keys())].rename(columns = selected_dict)
#
plt_res.index = pd.to_datetime(plt_res.index.astype('str'),format='%Y%m%d')
plt_rt = (plt_res + 1).cumprod()
plt_rt = plt_rt / plt_rt.iloc[0]
plt_rt_draw = plt_rt.iloc[-1].to_frame().T - 1
plt_rt_draw3=plt_rt_draw

#第2个月
for file in tqdm(files[old_second_index:old_first_index]):  #每月1号晚上九点更新，否则若第二天则：-1
    print(file)
    sty = pd.read_csv(sty_path + file)

    date = int(file.replace('-', '').split('.')[0])
    sty = sty.drop(columns = 'trade_date').set_index('security_id')
    sty.index = [i.split('.')[0] for i in sty.index]
    ret = open_pct.loc[date]  ##########

    sty = sty.apply(lambda x: (x - x.mean()) / x.std())
    sty = sty.apply(lambda x: x / x.abs().sum())
    daily_res = sty.apply(lambda x: (x * ret).sum(), axis = 0).rename(date).to_frame()
    res = pd.concat([res, daily_res], axis = 1)
plt_res = res.T
plt_res = plt_res[list(selected_dict.keys())].rename(columns = selected_dict)
plt_res.index = pd.to_datetime(plt_res.index.astype('str'),format='%Y%m%d')
plt_rt = (plt_res + 1).cumprod()
plt_rt = plt_rt / plt_rt.iloc[0]
plt_rt_draw = plt_rt.iloc[-1].to_frame().T - 1
plt_rt_draw2=plt_rt_draw


#第1个月
for file in tqdm(files[old_last_index:old_second_index]):  #每月1号晚上九点更新，否则若第二天则：-1
    print(file)
    sty = pd.read_csv(sty_path + file)

    date = int(file.replace('-', '').split('.')[0])
    sty = sty.drop(columns = 'trade_date').set_index('security_id')
    sty.index = [i.split('.')[0] for i in sty.index]
    ret = open_pct.loc[date]  ##########

    sty = sty.apply(lambda x: (x - x.mean()) / x.std())
    sty = sty.apply(lambda x: x / x.abs().sum())
    daily_res = sty.apply(lambda x: (x * ret).sum(), axis = 0).rename(date).to_frame()
    res = pd.concat([res, daily_res], axis = 1)
plt_res = res.T
plt_res = plt_res[list(selected_dict.keys())].rename(columns = selected_dict)
plt_res.index = pd.to_datetime(plt_res.index.astype('str'),format='%Y%m%d')
plt_rt = (plt_res + 1).cumprod()
plt_rt = plt_rt / plt_rt.iloc[0]
plt_rt_draw = plt_rt.iloc[-1].to_frame().T - 1
plt_rt_draw1=plt_rt_draw

###！！！修改列名和文件名
if True:
    barra=pd.concat([plt_rt_draw1,plt_rt_draw2,plt_rt_draw3],axis=0)
    barra.index=['M1','M2','M3']
    barra.to_csv(savepath+barra_name,encoding="utf_8_sig")
#

print('季度barra统计 finished')


#