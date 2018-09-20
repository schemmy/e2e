import numpy as np
import pandas as pd
import os
import warnings
from scipy.stats import gamma
import datetime as dt
from utils.demand_pkg import *


class data_parser:

    def __init__(self):
        pass

    def process_vlt_data(self, path, path_to, filename):

        file_to_save = filename.split('.')[0] + '_prep.csv'
        if  os.path.exists(path_to+file_to_save):
            df_vlt = pd.read_csv('%s%s' %(path_to, file_to_save), parse_dates=['create_tm','complete_dt','dt'])
            print('Vlt processed data read!')
        else:
            if not os.path.isdir(os.path.join(os.getcwd(), path_to)):
                os.makedirs(path_to)
            df_vlt = pd.read_csv('%s%s' %(path, filename), parse_dates=['create_tm','complete_dt','dt'])

            df_vlt['create_tm_index'] = (df_vlt['create_tm']- dt.datetime.strptime('2016-01-01','%Y-%m-%d')).dt.days
            df_vlt['complete_dt_index'] = (df_vlt['complete_dt']- dt.datetime.strptime('2016-01-01','%Y-%m-%d')).dt.days

            df_vlt = df_vlt.dropna(how='any')
            df_vlt = df_vlt.sort_values(['item_sku_id','int_org_num','create_tm_index','dt'], ascending=[True, True, True, True])

            df_vlt.insert(1, 'sku_id', df_vlt['item_sku_id'])
            df_vlt['item_sku_id'] = df_vlt[['item_sku_id', 'int_org_num']].astype(str).apply(lambda x: '#'.join(x), axis=1)
            df_vlt = df_vlt.drop_duplicates(['item_sku_id', 'create_tm_index'], keep='last')
            df_vlt = df_vlt.reset_index(drop=True)
            
            df_vlt.insert(1, 'sku_id', df_vlt['item_sku_id'])
            df_vlt.loc[:,'sku_id'] = o1['sku_id'].apply(lambda x: x.split('#')[0])
            
            df_vlt.to_csv('%s%s' %(path_to, file_to_save), index=False)
            print('Vlt raw data processed!')

        return df_vlt


    def process_sales_data(self, quantile_list, quantile_window_list, mean_window_list, path, path_to, filename):

        df_sl = pd.read_csv('%s%s' %(path, filename), index_col=0)
        df_sl.rename(columns=lambda x: (dt.datetime(2016,1,1) + dt.timedelta(days=int(x)-730)).date(), inplace=True)
        print('Sales data read!')

        get_rolling_mean(df_sl, mean_window_list, path=path_to)
        get_rolling_quantile(df_sl, quantile_list, quantile_window_list, path=path_to)

        print('Sales features generated!')
        return df_sl


    def read_sales_data(self, quantile_list, quantile_window_list, mean_window_list, path):

        quantile_feature, mean_feature = read_feature_data(quantile_list, quantile_window_list, mean_window_list, path)        
        print('Sales features read!')
        return quantile_feature, mean_feature


    def add_stock_feature(self, X, raw_path, path_to, stock_file, file_to_save):


        df_st = pd.read_csv('%s%s' %(raw_path, stock_file), index_col=0)
        df_st.columns = pd.to_datetime(df_st.columns)
        # X['create_tm'] = pd.to_datetime(X['create_tm'])
        # X['create_tm'] = pd.to_datetime(X['create_tm'])

        X['initial_stock'] = X.apply(lambda x: df_st.loc[x['item_sku_id'], x['create_tm'].normalize() ] \
                                if x['item_sku_id'] in df_st.index else np.nan, axis=1)
        X['VLT'] = (X['complete_dt'] - X['create_tm']) / timedelta (days=1)
        X['review_period'] = X['create_tm'].diff().shift(-1)/ timedelta (days=1)
        X.loc[X['review_period'] <=0, 'review_period'] = (dt.datetime(2018,8,31) - \
                                    X.loc[X['review_period'] <=0, 'create_tm'])/ timedelta (days=1)
        # X.to_csv('%s%s' %(path_to, file_to_save), index=False)
        print('Stock features aggregated!')
        return X


    def add_3b_feature(self, X, raw_path, path_to, stock_file, file_to_save):

        Z90 = 1.2816
        PERCENT = 0.9
        '''--------------------- Normal Basestock -----------------------'''
        sale_mean = X['mean_112']
        sale_std = X['std_140']
        VLT_mean = X['vendor_vlt_mean']
        VLT_std = 0
        T = X['review_period'] + X['vendor_vlt_mean']
        X['normal'] = (sale_mean*T + Z90*np.sqrt(T*sale_std**2+sale_std**2*VLT_std)-X['initial_stock']).fillna(0).clip(0)

        '''--------------------- Gamma Basestock -----------------------'''
        theta = sale_std**2/(sale_mean+0.0000001)
        k = sale_mean/(theta+0.0000001)
        k_sum = T*k
        X['gamma'] = pd.Series(gamma.ppf(PERCENT, a=k_sum, scale = theta)) - X['initial_stock']
        X['gamma'] = X['gamma'].fillna(0).clip(0)

        '''--------------------- Empirical Quantile Basestock -----------------------'''
        X['eq'] = (X['q_112'] * T - X['initial_stock']).clip(0)
        # X.to_csv('%s%s' %(path_to, file_to_save), index=False)
        print('3 brothers features aggregated!')
        return X



    def add_target(self, X, path, filename, path_to, file_to_save):

        df_sl = pd.read_csv('%s%s' %(path, filename), index_col=0)
        df_sl.rename(columns=lambda x: (dt.datetime(2016,1,1) + dt.timedelta(days=int(x)-730)).date(), inplace=True)

        X['next_complete_dt'] = X.groupby('item_sku_id').complete_dt.shift(-1).fillna(dt.datetime(2018,8,31))
        X['demand_RV'] = X.apply(lambda x: sum(df_sl.loc[x['item_sku_id'], \
                                            x['create_tm'].date():x['next_complete_dt'].date()].values)\
                            if x['item_sku_id'] in df_sl.index else np.nan, axis=1)

        X['demand_V'] = X.apply(lambda x: sum(df_sl.loc[x['item_sku_id'], \
                                            x['create_tm'].date():x['complete_dt'].date()].values)\
                            if x['item_sku_id'] in df_sl.index else np.nan, axis=1)

        X['target_decision'] = X['demand_RV'] - X['initial_stock']
        X['target_decision'].clip(0.0, inplace=True)

        X['target_decision_nobc'] = X['demand_RV'] - X['demand_V'] \
                                    - (X['initial_stock'] - X['demand_V']).clip(0.0)
        X['target_decision_nobc'].clip(0.0, inplace=True)

        X.to_csv('%s%s' %(path_to, file_to_save), index=False)
        print('Targets aggregated!')
        return X


    def get_vlt_sales_feature(self, quantile_list, quantile_window_list, mean_window_list, pred_len_list, 
                              raw_path, process_path, path_to,
                              vlt_file, filled_sale_file, file_to_save,
                              Model='M2Q', is_far=False):

        if os.path.exists('%s%s' %(path_to, file_to_save)):
            X = pd.read_csv('%s%s' %(path_to, file_to_save), parse_dates=['create_tm','complete_dt','dt'] )
            print('VLT and sales feature read!')

        else:
            df_vlt = self.process_vlt_data(raw_path, process_path, vlt_file)
            df_sl = self.process_sales_data(quantile_list, quantile_window_list, mean_window_list, 
                                            raw_path, process_path, filled_sale_file)
            quantile_feature, mean_feature = self.read_sales_data(quantile_list, quantile_window_list, mean_window_list, process_path)

            test_date = df_vlt['create_tm'].dt.normalize().rename('train_date')
            pred_len = pred_len_list[0]
            q = quantile_list[0]

            X = prepare_training_data(df_sl, q, 
                                     is_far, 
                                     pred_len, 
                                     Model, 
                                     test_date, 
                                     mean_window_list, 
                                     quantile_window_list, 
                                     quantile_feature,
                                     mean_feature,
                                     df_vlt['item_sku_id'],
                                     is_train=False)

            X = pd.concat([df_vlt, X], axis=1)
            X.to_csv('%s%s' %(path_to, file_to_save), index=False)

            print('VLT and sales feature obtained and aggregated!')
        return X

    def add_more_and_labels(self, X, raw_path, output_path, filled_sale_file, stock_file, file_to_save):

        X = self.add_stock_feature(X, raw_path, output_path, stock_file, file_to_save)
        X = self.add_3b_feature(X, raw_path, output_path, stock_file, file_to_save)
        X = self.add_target(X, raw_path, filled_sale_file, output_path, file_to_save)
        return X



raw_path = '../data/1320/'
process_path = '../data/1320/'
output_path = '../data/1320_feature/'

filled_sale_file = 'rdc_sales_1320_replenishment_V1_filled_pp.csv'
vlt_file = 'vlt_2018_0708.csv'
stock_file = 'stock.csv'
feature_file = 'features_v1.csv'
feature_file2 = 'features_v11.csv'

quantile_list = [90]
quantile_window_list = [7,14,28,56,112]
mean_window_list = [3,7,14,28,56,112]
pred_len_list = [31, 14, 7, 3, 1]

o = data_parser()
X = o.get_vlt_sales_feature(quantile_list, quantile_window_list, mean_window_list, pred_len_list,
                  raw_path, process_path, output_path, vlt_file, filled_sale_file, feature_file)
o.add_more_and_labels(X, raw_path, output_path, filled_sale_file, stock_file, feature_file2)



