import pandas as pd
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
# from alphalens.utils import get_clean_factor_and_forward_returns
import seaborn as sns
import datetime as dt
from datetime import datetime,timedelta,date
import os
import math
import matplotlib.pyplot as plt
import pickle as pkl
import statsmodels.api as sm
import warnings
import sys
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

class Environment():
    def __init__(self, factor_name, factor, holding_periods, save_path, env_pkl_path, method = 'create', if_neutral = None, Twap = True):
        self.factor_name = factor_name
        self.raw_factor = factor.copy()
        self.factor = factor.copy()    # index: int(date)  columns: str(6 number)
        self.holding_periods = holding_periods # [1, 5, 10, 20]
        self.save_path = save_path  # "/XX/XX/XX/"
        self.env_pkl_path = env_pkl_path # "/XX/XX/"

        self.method = method         # 'load' or something else
        self.if_neutral = if_neutral # ['indus', 'size'] or None

        self.index_name = None
        self.env_dict = None
        self.tot_dict = None
        self.Twap = Twap
        self.grouped_factor = {} # dict {[1, 11): dataFrame}, 10-highest

    def create_env(self):
        print("创建回测环境")
        # 创建环境数据字典，需要开盘价、收盘价、交易状态、行业信息等
        
        # 日收盘价
        p_close = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/adjclose.csv", index_col = 0, parse_dates = True)
        p_close.columns = [i[:6] for i in p_close.columns]
        # twap(复权)
        p_twap = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/twap_all.csv", index_col = 0, parse_dates = True)
        p_twap.columns = [x[:6] for x in p_twap.columns]
        # 复权因子
        adjfactor = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/ADJFACTOR.csv", index_col = 0, parse_dates = True)
        adjfactor = adjfactor.drop([x for x in adjfactor.columns if x[-2:]=='BJ'], axis=1)
        adjfactor.columns = [x[:6] for x in adjfactor.columns]
        # 复权twap
        p_twap, adjfactor = p_twap.align(adjfactor, join='inner')
        p_twap = adjfactor * p_twap
        
        # 日开盘价
        p_open0 = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/adjopen.csv", index_col = 0, parse_dates = True) 
        p_open0.columns = [i[:6] for i in p_open0.columns]
        
        # 定义交易价格
        if self.Twap:
            p_price = p_twap
        else:
            p_price = p_open0

        # 使用前一日的收盘价填充交易价格为空的值
        p_price[p_price.isnull()] = p_close.shift()[p_price.isnull()]
        # 交易状态
        trade_status_all = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/trade_status_st_suspend_amount.csv", index_col = 0, parse_dates = True)
        trade_status_all.columns = [i[:6] for i in trade_status_all.columns]
        # 涨跌停
        up_limit = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/limit.csv", index_col = 0, parse_dates = True)
        up_limit.columns = [i[:6] for i in up_limit.columns]
        down_limit = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/stopping.csv", index_col = 0, parse_dates = True)
        down_limit.columns = [i[:6] for i in down_limit.columns]
        # 交易状态
        trade_status = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/trade_status.csv", index_col = 0, parse_dates = True)
        # st股票
        st_status = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/st_status.csv", index_col = 0, parse_dates = True)
        # 当日停牌
        suspend_status = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/suspend_status.csv", index_col = 0, parse_dates = True)
        # 统一格式
        suspend_status.columns = [i[:6] for i in suspend_status.columns]
        trade_status.columns = [i[:6] for i in trade_status.columns]
        st_status.columns = [i[:6] for i in st_status.columns]
        # 新股
        newstock = p_open0.isnull().astype('int').rolling(60, min_periods=1).max() == 1
        # 计算不同期限的未来收益率
        future_retd = {}
        for period in self.holding_periods:
            future_retd[period] = p_price.pct_change(period).shift(-period - 1).dropna(how='all')
        invalid_data = True
        # 剔除掉不能交易，新上市这些股票会被剔出股票池
        # 筛选开盘涨跌停的股票，这些股票将不参与交易
        if self.index_name==None:
            invalid_data = (suspend_status == True) | (st_status == True)
            limit_data =  (p_open0==up_limit) | (p_open0==down_limit)
        elif self.index_name in ['中证500','上证50','沪深300','国证a指']:
            # 只算指数成分股
            # 不完整，这部分功能还不能用，先把index_name 置为None
            index_member = hd.load_support_data(self.index_name+'成份_ts').drop_duplicates(subset=['date','code'])
            index_member_df = index_member.pivot(index='date',columns='code',values='rank')
            index_member_df.index = pd.to_datetime(index_member_df.index.to_series().astype(str)).apply(lambda x:x.to_pydatetime().date())
            index_member_df.columns = [x[2:] for x in index_member_df.columns]
            index_member_df = index_member_df.reindex(p_price.index).T.reindex(p_price.columns).T.fillna(0)
            p_price[index_member_df==0] = np.nan
            invalid_data = (trade_status == 'False') | (st_status == 'True') | newstock  | (trade_status_all == False)
            limit_data =  (p_open0==up_limit) | (p_open0==down_limit)

        # 初始化行业市值数据
        industry_index_vert = pd.DataFrame()
        size = pd.DataFrame()
        
        # 行业数据
        industry = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/zxIndustry_new_chinese.csv", encoding = 'gbk', index_col = 0, low_memory=False,  parse_dates = True)
        industry.columns = [i.split('.')[0] for i in industry.columns]
        industry = industry.stack().to_frame()
        industry_index_vert = pd.get_dummies(industry)
        # 市值数据  
        size = pd.read_csv("/data/disk2/yzhang/ricequant_barra/size.csv", index_col = 0, parse_dates = True)
        size = size.stack().to_frame()
        size.columns = ['size']
        # 行业市值组合
        indus_size = pd.concat([size, industry_index_vert], axis = 1).dropna()
        # 指数价格
        aindex = pd.read_csv("/data/disk1/data/DataBase_stock/AllSample/zz500_index.csv", index_col = 0, parse_dates = True).close
        self.env_dict = {'price':p_price, 'close': p_close, 'invalid':invalid_data, 'limit' : limit_data, 'future_retd':future_retd, 'index': aindex,'indus_size': indus_size, 'indus': industry_index_vert, 'size': size}

    def load_env(self):
        print('加载回测环境')
        os.chdir(self.env_pkl_path)
        if self.index_name==None:
            with open('env_dict.pkl','rb') as f:
                self.env_dict = pkl.load(f)
        else:
            with open('env_dict'+self.index_name+'.pkl','rb') as f:
                self.env_dict = pkl.load(f)

    def save_env(self):
        print('保存回测环境')
        os.chdir(self.env_pkl_path)
        if self.index_name==None:
            with open('env_dict.pkl','wb') as f:
                pkl.dump(self.env_dict,f)
        else:
            with open('env_dict'+self.index_name+'.pkl','wb') as f:
                pkl.dump(self.env_dict,f)

    def winsorize(self):
        '''将因子值进行极端值缩尾处理，拉回至3.5倍MAD水平，并且不影响排序，不减少覆盖度'''
        '''建议先不进行winsorize，观测因子的分布情况'''
        df = self.factor
        md = df.median(axis=1)
        mad = (1.483 * (df.sub(md, axis=0)).abs().median(axis=1)).replace(0,np.nan)
        up = df.apply(lambda k: k > md + mad * 3)
        down = df.apply(lambda k: k < md - mad * 3)
        df[up] = df[up].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md + mad * 3, axis=0)
        df[down] = df[down].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md - mad * (0.5 + 3), axis=0)
        self.factor = df
  
    def neutralized(x, item):
        sig = item[1]
        sig = sig.replace(0, np.nan).dropna(axis=1, how='all').replace(np.nan, 0)
        X = sig.iloc[:,1:].values
        y = sig.iloc[:,:1].values.ravel()

        X=X.astype(float)
        y=y.astype(float)
        
        beta = np.linalg.inv((X.T).dot(X))
        beta = beta.dot(X.T).dot(y)
        ans = y - X.dot(beta)
        item[1]['fac'] = ans
        return item[1].fac.to_frame()
    
    def neutralization(self, df):  
        print("行业市值中性化")

        if self.if_neutral == None:
            return self.factor
        if ('indus' not in self.if_neutral) & ('size' not in self.if_neutral):
            return self.factor
        
        if "indus" in self.if_neutral:
            indus_size = self.env_dict['indus']
        
        if 'size' in self.if_neutral:
            indus_size = self.env_dict['size']
        
        if ('size' in self.if_neutral) & ("indus" in self.if_neutral):
            indus_size = self.env_dict['indus_size']
        
        df.columns = [i.split('.')[0] for i in df.columns]
        df = df.stack().to_frame()
        df.columns = ['sig']
        
        final = pd.concat([df, indus_size], axis = 1)
        final = final.replace([np.inf, -np.inf], np.nan)
        final = final.dropna()
        grps = final.groupby(level=0)
        pool = Pool(15)
        ans = pool.map(self.neutralized, grps)
        pool.close()  # 进程池关闭
        pool.join()  # 进程池关闭的检测   
        ans = pd.concat(ans, axis=0)
        return ans.fac.unstack()
    
    def normalize(self, df):
        return df.apply(lambda x: (x - x.mean()) / x.std(), axis = 1)

    def process_factor(self):
        print('清洗因子数据并匹配收益率')
        self.format_factor()
        
        if 1 not in self.holding_periods: self.holding_periods=[1] + self.holding_periods
        self.holding_periods = sorted(self.holding_periods)
        
        # 判断是否存在inf
        if np.isinf(self.factor).any().any():
            print("Error: ", self.factor_name, ' 存在infinity.')
            sys.exit()
       
        self.factor = self.factor.dropna(how='all').reindex(self.env_dict['price'].index).T.reindex(self.env_dict['price'].columns).T.dropna(how='all')
        
        # 计算股票池中有因子值的占比
        out = self.env_dict['invalid'].fillna(True)
        cover_series = self.factor[out==False].count(axis=1) / (out==False).sum(axis=1)
        
        # 剔除非股票池中的数据
        out, self.factor = out.align(self.factor, join = 'inner')
        self.factor[out] = np.nan

        # 进行缩尾，标准化，中性化
        self.winsorize()
        self.factor = self.normalize(self.factor)
        if self.if_neutral==None:
            self.factor = self.factor
        else:
            self.factor = self.neutralization(self.factor)
            
        self.factor = self.normalize(self.factor)
        print('计算不同期限的未来收益率')
        future_retd_new = {}
        for period in self.holding_periods:
            if period in self.env_dict['future_retd'].keys():
                future_retd_new[period] = self.env_dict['future_retd'][period]
            else:
                future_retd_new[period] = self.env_dict['price'].pct_change(period, fill_method=None).shift(-period - 1).dropna(how='all')
        
        future_retd_new[self.holding_periods[-1]], self.factor = future_retd_new[self.holding_periods[-1]].align(self.factor, join = 'inner')

        self.tot_dict = {
                            'future_retd': future_retd_new,
                            'holding_periods': self.holding_periods,
                            'cover': cover_series}

    def format_factor(self):
        print("加载因子数据")
        factor = self.factor
        '''将因子格式调整一致'''
        try:
            c0 = factor.columns[0]
            type0 = type(c0)
            if type(c0)==str:
                if c0[0]=='S' or c0[0]=='s':
                    factor.columns = [x[2:] for x in factor.columns]
                else:
                    factor.columns = [format(int(x[:6]),'06d') for x in factor.columns]
            else:
                factor.columns = [format(int(x),'06d') for x in factor.columns]
        except:
            print('因子 columns 格式有误，请调整为"000001"字符串格式')
        
        try:
            i0 = factor.index[0]
            if type(i0) == int or type(i0) == float or type(i0) == np.int64:
                factor.index = [datetime.strptime(str(int(x)),'%Y%m%d') for x in factor.index]
            elif type(i0)==str:
                try:
                    factor.index = [datetime.strptime(str(int(x[:8])),'%Y%m%d') for x in factor.index]
                except:
                    try:
                        factor.index = [datetime.strptime(x[:10].strip(),'%Y-%m-%d') for x in factor.index]
                    except:
                        try:
                            factor.index = [datetime.strptime(x[:10].strip(),'%Y %m %d') for x in factor.index]
                        except:
                            print('因子 index 格式有误，请调整为 datetime.date 格式')

            elif type(i0)==datetime:
                factor.index = [x for x in factor.index]
            elif type(i0)==pd.Timestamp:
                factor.index = [x.to_pydatetime() for x in factor.index]
            elif type(i0)==date:
                pass
            else:
                print('因子 index 格式有误，请调整为 datetime.date 格式，现为',type(i0),i0)
            
        except:
            print('因子 index 格式有误，请调整为 datetime.date 格式，现为')

        self.factor = factor
        self.raw_factor = self.factor.copy()

    def factor_grouped(self):
        print("因子分组")
        group = round((self.factor.rank(pct=True,axis=1)*0.9999*10+0.5))
        for i in range(1, 11):
            self.grouped_factor[i] = self.factor[group == i]
        return

    def run_all(self):
        if self.method == 'load':
            self.load_env()
            self.save_env()
        else:
            self.create_env()
            self.save_env()
        self.process_factor()
        self.factor_grouped()

class Functions():
    def __init__(self, env):
        print('env success')
        self.env = env
        self.group_ret = {}
        self.ret_dict = {}
        # 保存路径为env.save_path + env.factor_name
        os.makedirs(env.save_path + env.factor_name, exist_ok = True)
    
    def factor_grouped_performance(self):
        print("因子分组表现")
        plt.clf()
        grouped_factor = self.env.grouped_factor
        n = len(grouped_factor) + 1
        future_retd = self.env.tot_dict['future_retd'][1]
        for period in self.env.holding_periods:
            accum_ret = {}
            annual_ret = {}
            volatility = {}
            start_date = {}
            end_date = {}
            self.group_ret[period] = pd.DataFrame()
            for i in range(1,n):
                # 计算仓位
                position = pd.DataFrame(np.where(grouped_factor[i].isnull(),0,1), index=grouped_factor[i].index, columns=grouped_factor[i].columns)
                position[self.env.env_dict['limit'].shift(-1) == True] = np.nan # 剔除第二日的涨跌停股票
                position = position.div(position.sum(axis=1), axis=0)
                position = position.rolling(window=period).mean()
                position, future_retd = position.align(future_retd, join='inner')
                #计算持仓每日的收益率
                dret = (position * future_retd).sum(axis=1)
                self.group_ret[period][i] = dret
                # 累计收益
                accum_ret[i] = np.prod(dret+1) - 1
                # 年化收益
                annual_ret[i] = np.power(accum_ret[i]+1, 252/dret.count()) - 1
                #年化波动
                volatility[i] = dret.rolling(window=252, min_periods=200).std().mean() * np.sqrt(252)
                start_date[i] = list(dret.index.date)[0]
                end_date[i] = list(dret.index.date)[-1]

            n_v = (self.group_ret[period] + 1).cumprod()
            n_v['market_nv'] = (self.env.env_dict['index'].pct_change() + 1).cumprod().loc[n_v.index]
            n_v['market_nv'] = n_v['market_nv'] / n_v['market_nv'].iloc[0]
            plt.figure()
            n_v.plot(colormap='nipy_spectral')
            plt.title('十组净值变化'+ str(period).zfill(2) + "天")
            plt.savefig(self.env.save_path + self.env.factor_name + '/因子分组净值-' + str(period) +'.png')
            
            start_date = list(start_date.values())
            end_date = list(end_date.values())      
            acr = ['{:.2%}'.format(x) for x in list(accum_ret.values())]
            ar = ['{:.2%}'.format(x) for x in list(annual_ret.values())]
            vol = ['{:.2%}'.format(x) for x in list(volatility.values())]
            col = ['group'+str(i) for i in range(1,n)]
            row = ['startdate', "end_date","accumulated return", "annual_return", "volatility"]
            plt.figure()
            tab = plt.table(cellText=np.vstack((start_date, end_date,acr,ar,vol)),
                            colLabels=col,
                            rowLabels=row,
                            loc='center', 
                            cellLoc='center',
                            rowLoc='center')
            tab.scale(3,2) 
            plt.axis('off')
            
            plt.title("十组收益--持仓" + str(period).zfill(2) + "天", y = 0.8)
            
            plt.savefig(self.env.save_path + self.env.factor_name + '/因子表现-' + str(period) +'.png', dpi=70, bbox_inches='tight')
    
    def factor_grouped_returns(self):
        print("因子分组收益图")
        plt.clf()
        pic_num = len(self.env.holding_periods)
        row_col = pow(pic_num ,0.5)
        if row_col == int(row_col):
            row_col = int(row_col)
        else:
            row_col = int(row_col) + 1

        fig = plt.figure(figsize=(18,8))  #设置一个图像长宽 
        ax_list = [fig.add_subplot(row_col, row_col, i) for i in range(1, pic_num + 1)]

        for period, ax_t in zip(self.env.holding_periods, ax_list):
            group_ret = self.group_ret[period].copy()
            group_ret['year'] = [str(i) for i in group_ret.index.year]
            ret = group_ret.groupby(by='year').apply(lambda x: np.prod(1+x) - 1)
            ret.plot(ax = ax_t, kind='bar', title = '持仓期-' + str(period))
        
        for i in range(pic_num):
            if (i + 1) % row_col == 0:
                ax_list[i].legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
            else:
                ax_list[i].legend_.remove()
        
        fig.tight_layout()
        fig.savefig(self.env.save_path + self.env.factor_name + '/因子分组收益图.png', dpi=70, bbox_inches='tight')
        
    def factor_netvalue(self):
        print("因子净值曲线")
        plt.clf()
        adj_close = self.env.env_dict['close']
        adj_twap = self.env.env_dict['price']
        # 获取多空的仓位 （只考虑打分最高和最低分组）
        short = self.env.grouped_factor[1].copy()
        long = self.env.grouped_factor[10].copy()
        short = short.apply(lambda x: x / x.abs().sum(), axis = 1)
        long = long.apply(lambda x: x / x.abs().sum(), axis = 1)
        # 剔除开盘涨跌停的股票
        short[self.env.env_dict['limit'].shift(-1)==True] = np.nan
        long[self.env.env_dict['limit'].shift(-1)==True] = np.nan
        # 持仓收益
        short = short.shift(1).dropna(how='all').fillna(0) # 使用0填充空值
        long = long.shift(1).dropna(how='all').fillna(0) # 使用0填充空值
        short_hold_returns_1 = (short.shift(1) * (adj_close/adj_close.shift(1) - 1 )).dropna(how='all').sum(axis = 1)
        long_hold_returns_1 = (long.shift(1) * (adj_close/adj_close.shift(1) - 1 )).dropna(how='all').sum(axis = 1)
        # 新增收益
        short_run_returns_1 = ((short - short.shift(1)) * (adj_close/adj_twap - 1)).dropna(how='all').sum(axis = 1)
        long_run_returns_1 = ((long - long.shift(1)) * (adj_close/adj_twap - 1)).dropna(how='all').sum(axis = 1)
        short_return = (short_hold_returns_1 + short_run_returns_1).dropna(how='all').rename('short')
        long_return = (long_hold_returns_1 + long_run_returns_1).dropna(how='all').rename('long')
        # 交易成本
        buy_sell = (long - long.shift(1)).dropna(how='all')
        # buy_side 万5
        cost_buy = buy_sell[buy_sell > 0].sum(axis = 1) * 5e-4
        # sell_side 千1.5
        cost_sell = buy_sell[buy_sell < 0].abs().sum(axis = 1) * 1.5e-3
        cost_1 = cost_buy.fillna(0) + cost_sell.fillna(0)
        # 多空收益
        long_short_return =  (short_return / 2 +  long_return / 2).rename('long_short')
        # universe收益
        universe = self.env.factor.apply(lambda x: x / x.abs().sum(), axis = 1)
        universe_return = (universe * self.env.tot_dict['future_retd'][1]).dropna(how = 'all').sum(axis = 1).rename('universe')
        # benchmark 收益
        index_return = self.env.env_dict['index'].pct_change().rename('benchmark')
        # 超额收益（不含交易成本）
        extra_return = (long_return  - index_return).rename('extra')
        # 
        extra_return_2 = (long_return  - index_return - cost_1).rename('extra_with_cost')
        # 存储收益数据
        self.ret_dict = {'long_ret': long_return, 'short_ret': short_return, 'long_short_ret': long_short_return, 'universe_ret': universe_return, 'index_ret': index_return, 'extra_ret':extra_return, 'extra_ret_withcost': extra_return_2}
        # 净值变化
        all_return = pd.concat([short_return, long_return, long_short_return, universe_return, index_return, extra_return, extra_return_2], axis = 1).dropna(how = 'any')
        all_nv = (all_return + 1).cumprod()

        all_nv = all_nv/ all_nv.iloc[0]
        all_nv.plot(title = 'Net Value')
        plt.savefig(self.env.save_path + self.env.factor_name + '/因子净值曲线.png', dpi=150,)
        all_return['year'] = [str(i) for i in all_return.index.year]
        byyear_return = all_return.groupby(by='year').apply(lambda x: (x+1).cumprod())
        for i in all_return.year.unique():
            cur = byyear_return.loc[i]
            cur.plot(title = 'by_year_Net_Value'+i)
            plt.savefig(self.env.save_path + self.env.factor_name + "/"+i+'_因子净值曲线.png', dpi=150,)
        return
    #算最大回撤
    def get_max_drawdown_fast(self, array):
        """
        传入net value的序列
        """
        drawdowns = {}
        duration = {}
        max_so_far = array.iloc[0]
        max_date = array.index[0]
        for i in array.index:
            if array.loc[i] > max_so_far:
                drawdown = 0
                drawdowns[i] = drawdown
                duration[i] = 'None'
                max_so_far = array.loc[i]
                max_date = i
            else:
                drawdown = max_so_far - array.loc[i]
                drawdowns[i] = drawdown
                duration[i] = str(max_date)[:10] + '-' + str(i)[:10]
        mdd_date = max(drawdowns, key=lambda x: drawdowns[x])
        return drawdowns[mdd_date], duration[mdd_date]
    
    def factor_long_short_performance(self):
        print("因子多空表现")
        all_return = pd.concat([self.ret_dict['short_ret'], self.ret_dict['long_ret'], self.ret_dict['long_short_ret'], self.ret_dict['universe_ret'], self.ret_dict['index_ret'], self.ret_dict['extra_ret']], axis = 1).dropna()
        all_nv = (all_return + 1).cumprod()

        annual_ret = pow(all_nv.iloc[-1], 252/len(all_nv)).rename('Annualized Return') - 1
        annual_std = (all_return.std() * np.sqrt(252)).rename("Annualized Volatility")
        annual_sharp = ((annual_ret - annual_ret.benchmark) / annual_std).rename("Annualized Sharp")
        dd = all_nv.apply(lambda x: self.get_max_drawdown_fast(x))
        max_drawdown = dd.iloc[0].rename('max_drawdown')
        max_drawdown_date = dd.iloc[1].rename('max_drawdown_duration')
        # max_drawdown = all_return.min().rename('Maximum Drawdown')
        # max_drawdown_date = all_return.idxmin().apply(lambda x: str(x).replace('-','')[:8]).rename('Maximum Drawdown Date')
        temp = pd.concat([annual_ret, annual_std, annual_sharp, max_drawdown, max_drawdown_date],axis = 1).T

        plt.figure()
        tab = plt.table(cellText=np.vstack(temp.values),
                    colLabels=temp.columns,
                    rowLabels=temp.index,
                    loc='center', 
                    cellLoc='center',
                    rowLoc='center')
        tab.scale(3,2) 
        plt.axis('off')
        plt.title("因子多空表现", y = 0.8)
        plt.savefig(self.env.save_path + self.env.factor_name + '/因子多空表现.png', dpi=70, bbox_inches='tight')

    def factor_distribution(self):
        plt.clf()
        print("因子分布")
        factor_value = self.env.raw_factor
        f = factor_value.stack().reset_index()
        f['year'] = f.level_0.apply(lambda x: x.year)
        allyear = f.year.unique()
        n = len(allyear) + 1
        
        if n % 3 == 0:
            ncol = int(n/3)
        else:
            ncol = int(n/3) + 1
        fig, axes = plt.subplots(3, ncol, figsize = (10,7), sharex = True, sharey = True)

        for i in range(n-1):
            if ncol == 1:
                ax = sns.kdeplot(f.loc[f.year == allyear[i], 0], color = 'g', fill = True, ax = axes[int(i/ncol)], label = allyear[i])
            else:
                ax = sns.kdeplot(f.loc[f.year == allyear[i], 0], color = 'g', fill = True, ax = axes[int(i/ncol), i%ncol], label = allyear[i])
            ax.legend(loc="upper right")
        i += 1
        if ncol == 1:
            ax = sns.kdeplot(f[0], color = 'g', fill = True, ax = axes[int(i/ncol)], label = 'all_period')
        else:
            ax = sns.kdeplot(f[0], color = 'g', fill = True, ax = axes[int(i/ncol), i%ncol], label = 'all_period')
        ax.legend(loc="upper right")
        fig.get_figure().savefig(self.env.save_path + self.env.factor_name + '/因子分布.png', dpi=100,)
    
    def factor_coverage(self):
        print("因子覆盖度")
        plt.clf()
        self.env.tot_dict['cover'].plot(title='cover')
        plt.savefig(self.env.save_path + self.env.factor_name + '/因子覆盖度.png', dpi=100,)

    def ic_calculating(self):
        print('因子IC分析')
        ic_list = []
        ir_list = []
        
        for period in self.env.tot_dict['holding_periods']:
            ic_series = self.env.factor.apply(lambda x:x.corr(self.env.tot_dict['future_retd'][period].loc[x.name], method='spearman'), axis=1)
            ic_list.append(ic_series.mean())
            ir_list.append(ic_series.mean()/ic_series.std())
        by_year_ic = self.env.factor.apply(lambda x:x.corr(self.env.tot_dict['future_retd'][1].loc[x.name], method='spearman'), axis=1).to_frame()
        by_year_ic = by_year_ic.set_index([by_year_ic.index.year, by_year_ic.index]).groupby(level=0).mean()
        by_year_ic.columns = ['IC']
        by_year_ic.index = by_year_ic.index.astype('str')
        by_year_ic.plot(title='IC')
        print(by_year_ic)
        plt.savefig(self.env.save_path + self.env.factor_name + '/因子逐年IC.png', dpi=110)
        col = self.env.tot_dict['holding_periods']
        row = ["IC", "ICIR"]
        plt.figure(figsize=(10,2))
        tab = plt.table(cellText=np.vstack((ic_list,ir_list)),
                        colLabels=col,
                        rowLabels=row,
                        loc='center', 
                        cellLoc='center',
                        rowLoc='center')
        tab.scale(1,2) 
        plt.axis('off')
        plt.axis('off')
        plt.title("因子IC分析", y = 0.8)
        plt.savefig(self.env.save_path + self.env.factor_name + '/因子IC分析.png', dpi=70, bbox_inches='tight')

    def run_all(self):
        self.factor_grouped_performance()
        self.factor_grouped_returns()
        self.factor_netvalue()
        self.factor_distribution()
        self.factor_coverage()
        self.factor_long_short_performance()
        self.ic_calculating()
        
class Performance():
    def __init__(self, save_path):
        self.save_path = save_path
        pass
        
    def render(self):
        print("合并为PDF")
        # 保存至save_path
        imgs = [os.path.join(self.save_path, name) for name in sorted(os.listdir(self.save_path)) if name[-3:] == 'png']
        size_list, img_list, height = [], [], 0
        for im in imgs:
            # try:
            img = Image.open(im).convert('RGB')
            print(img.size, im.split('/')[-1])
            img = img.resize((1000, int(img.height * 1000 / img.width)))
            img_list.append(img)
            size_list.append(list(img.size))
            
            height += img.size[1] + 5
            # except:
            #     pass

        
        size_lt = sorted(size_list, key=lambda x: x[0])
        width = size_lt[-1][0] + 10
        canv = Image.new('RGB', (width, height), (255, 255, 255))
        y = 10
        for i in range(len(img_list)):
            size = size_list[i]
            dx = (width - size[0]) // 2
            canv.paste(img_list[i], (dx, y))
            img_list[i].close() # 用完就顺手关闭一下图片
            y += size[1] + 5
        draw = ImageDraw.Draw(canv)
        draw.rectangle((0, 0, canv.size[0]-1, canv.size[1]-1), outline='black', width=5)
        canv.save(os.path.join(self.save_path, "Overall_Performance.pdf"), 'PDF', resolution=100.0, save_all=True)


if __name__=='__main__':
    factor_path = "/data/disk4/output_stocks/jmchen/factors/shot/spread/entire_files/z10_vwap.csv"
    factor_name = 'z10_vwap'
    result_path = "/data/disk4/output_stocks/jmchen/factors/shot/spread/performance/"
    env_path = "/data/disk4/output_stocks/jmchen/backtool/"
    os.makedirs(result_path, exist_ok = True)

    factor = pd.read_csv(factor_path, index_col=0)
    factor.index = pd.to_datetime(factor.index.astype('str'))
    factor = factor.replace([np.inf,-np.inf], np.nan)

    # factor1 = pd.read_csv('/data/disk4/output_stocks/jmchen/factors/transaction/mid_files/bid_mean.csv', index_col=0)
    # factor1.index = pd.to_datetime(factor1.index.astype('str'))
    # factor1 = factor1.replace([np.inf,-np.inf], np.nan)

    # factor=(factor/factor1)

    #factor=factor.replace(0,np.nan)
    #factor = factor.rolling(window=20, min_periods=15).mean()
    # factor = factor.loc['2016-01-01':'2022-01-01']
    env = Environment(
                        factor_name = factor_name,
                        factor = -factor,
                        holding_periods = [1,5,10,20], 
                        save_path = result_path, 
                        env_pkl_path = env_path,
                        method='load',
                        if_neutral=None
                        #if_neutral=['indus','size']
                        )
    env.run_all()

    funs = Functions(env)
    funs.ic_calculating()
    funs.run_all()
    perform = Performance(result_path + factor_name)
    perform.render()


 