# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:04:49
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-25 16:54:28

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import copy,os
import pickle
from datetime import timedelta
import sys
sys.path.append('../')
from utils.benchmark import *

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# ## Experiment part 1: End to end V.S. two-stage model (MQRNN)
b, h = 9, 1

def get_bm2(x):
    rl = np.ceil(x['review_period']) + np.ceil(x['E2E_NN_vlt_pred'])
    if rl <= b:
        days = int(rl)
    else:
        days = int(rl) - rl//(b+h)
    return x['Bm2_pred'] * days


df_sales = pd.read_csv('../data/1320/rdc_sales_1320_replenishment_V1_filled_pp.csv')
df_sl = df_sales.set_index('row')
df_sl.rename(columns=lambda x: (dt.datetime(2016,1,1) + dt.timedelta(days=int(x)-730)).date(), inplace=True)
df_sl['SKU'], df_sl['DC'] = df_sl.index.str.split('#', 1).str


START_DAY = dt.datetime(2018,7,27)

#Benchmark2 qunatile prediction: quantiles = [0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
with open('../logs/torch/pred_E2E_SF_RNN.pkl', 'rb') as fp: 
    bc2_o = pickle.load(fp)

#Item_sku_id, sku_id, create_tm, complete_dt, vlt_actual, E2E_MLP_pred, E2E_NN_SF_mean_pred, E2E_NN_vlt_pred 
df_orders_o = pd.read_csv('../seq2seq/pred_v5.csv', parse_dates=['create_tm','complete_dt','next_complete_dt'])


for i in [5, 10]:
    st = i*500
    ed = (i+1)*500
    bc2 = bc2_o[st:ed,:,:]
    df_orders = df_orders_o.iloc[st:ed,:]

    o4 = df_orders.copy()

    list_c2 = ['SKU_DC',
              'E2E_MLP', 'E2E_RNN',
              'Bm1','Bm2',  
              'Ave_sales'
             ]
    str_list2 = ['E2E_MLP', 'E2E_RNN', 'Bm1','Bm2']


    o4['demand_RV_list'] = o4.apply(lambda x: df_sl.loc[x['item_sku_id'],
                                x['create_tm'].date():x['next_complete_dt'].date()].values
                             if x['item_sku_id'] in df_sl.index else [], axis=1)
    o4['demand_RV_list_acm'] = o4['demand_RV_list'].apply(lambda x: np.cumsum(x))


    o4g = o4.groupby('item_sku_id').agg(lambda x: x.tolist())
    o4g['Demand_agg_list'] = o4g.apply(lambda x: df_sl.loc[x.name,
                                x['create_tm'][0].date():x['next_complete_dt'][-1].date()].values,
                                axis=1)
    numberOfRows = len(o4g)
    o4g = o4g.reset_index(drop=True)

    o4g['E2E_MLP_agginv_f'], o4g['E2E_MLP_agginv'] = zip(*o4g.apply(get_agginv, name='E2E_MLP',  axis=1))
    o4g['E2E_RNN_agginv_f'], o4g['E2E_RNN_agginv'] = zip(*o4g.apply(get_agginv, name='E2E_RNN',  axis=1))

    table = np.zeros((6,4))
    for p in range(2,8):
        o4['Bm1_pred'] = np.mean(bc2[:,:,p], axis=1)
        o4['Bm1_pred'] = o4['Bm1_pred'] * (np.ceil(o4['review_period']) + np.ceil(o4['E2E_NN_vlt_pred']))

        o4['Bm2_pred'] = np.mean(bc2[:,:,p], axis=1)
        o4['Bm2_pred'] = o4.apply(get_bm2, axis=1)

        o4['Bm1_replen'] = (o4['Bm1_pred'] - o4['initial_stock']).clip(0)
        o4['Bm2_replen'] = (o4['Bm2_pred'] - o4['initial_stock']).clip(0)

        o4g[['Bm1_pred','Bm2_pred']] = o4[['item_sku_id','Bm1_pred','Bm2_pred']]\
                            .groupby('item_sku_id').agg(lambda x: x.tolist()).reset_index(drop=True)

        o4g['Bm1_agginv_f'], o4g['Bm1_agginv'] = zip(*o4g.apply(get_agginv, name='Bm1',  axis=1))
        o4g['Bm2_agginv_f'], o4g['Bm2_agginv'] = zip(*o4g.apply(get_agginv, name='Bm2',  axis=1))


        df2_cost_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=list_c2)
        df2_holding_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=list_c2)
        df2_back_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=list_c2)
        df2_stockout_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=list_c2)
        df2_turnover_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=list_c2)

        df2_cost_agg['SKU_DC']=df2_holding_agg['SKU_DC']=df2_back_agg['SKU_DC']=df2_stockout_agg['SKU_DC']=df2_turnover_agg['SKU_DC']=o4g.index.values
        df2_cost_agg['Ave_sales']=df2_holding_agg['Ave_sales']=df2_back_agg['Ave_sales']=df2_stockout_agg['Ave_sales']=df2_turnover_agg['Ave_sales']=o4g['mean_14'].apply(lambda x:x[0]).values

        for str1 in str_list2:
            str2 = str1 + '_agginv'
            df2_holding_agg[str1] = o4g[str2].apply(lambda x: h * sum([inv for inv in x if inv>0]) )
            df2_back_agg[str1] = o4g[str2].apply(lambda x: b * -sum([inv for inv in x if inv<0]) )
            df2_stockout_agg[str1] = o4g[str2].apply(lambda x: len([inv for inv in x if inv<0])/len(x) if len(x)>0 else 0 )
            df2_turnover_agg[str1] = o4g.apply(lambda x: np.mean([max(i,0) for i in x[str2]]) / x['mean_14'][0]
                                          if np.mean(x['mean_14'][0]) >0 else np.mean(x[str2]), axis=1).fillna(7)
            df2_cost_agg[str1] = df2_holding_agg[str1] + df2_back_agg[str1]

        df2_aggcom = pd.DataFrame({'Total cost': df2_cost_agg[str_list2].mean(),
                     'Holding cost': df2_holding_agg[str_list2].mean(),
                     'Stockout cost': df2_back_agg[str_list2].mean(),
                     'Stockout rate': df2_stockout_agg[str_list2].mean(),
                     'Turnover rate': df2_turnover_agg[str_list2].mean(),
                     }).T

        table[p-2, 0] = df2_aggcom.iloc[0, 0]
        table[p-2, 1] = df2_aggcom.iloc[0, 1]
        table[p-2, 2] = df2_aggcom.iloc[0, 2]
        table[p-2, 3] = df2_aggcom.iloc[0, 3]


    df_cost_agg = pd.DataFrame(table, columns=str_list2, index=['50','60','70','80','90','95'])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df_cost_agg.iloc[:,0], c='C0');
    ax.plot(df_cost_agg.iloc[:,1], c='C1');
    ax.plot(df_cost_agg.iloc[:,2], c='C2');
    ax.plot(df_cost_agg.iloc[:,3], c='C3');
    ax.legend(df_cost_agg.columns, loc=0)
    ax.set_xlabel('Quantitle of demand forecasting')
    ax.set_ylabel('Total Cost')
    ax.set_ylim((3500,9000))
    ax.set_title('Comparison with two benchmarks')

    plt.savefig('../figures/eps/step_%i.eps' %i,dpi=200);

# print(df2_aggcom.to_latex(float_format=lambda x: '%.2f' % x))

