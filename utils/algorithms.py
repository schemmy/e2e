#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:05:56 2018

@author: yyshi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

def get_opt(df_opt, inv, t, sku, dc):
    df = df_opt[df_opt.sku_id == sku]
    df = df[df.int_org_num == dc]
    #opt = df['overall_opt_order'].iloc[t] 
    opt = df['overall_opt_order'].iloc[t] 
    return opt

def get_EQ(record, inv, t, renew_period, PERCENT=90):
    sales = record[0:t]
    EQ_series = [np.sum(sales[i:i+renew_period]) for i in range(len(sales)-renew_period)]
    EQ_value = np.percentile(EQ_series, PERCENT)
    action = np.max([EQ_value - inv, 0])
    return int(action)

def get_gamma_basestock(record, inv, t, renew_period, PERCENT=0.9):
    sales = record[0:t]
    mean = np.mean(sales)
    var = np.var(sales)
    theta = var/(mean+0.0000001)
    k = mean/(theta+0.0000001)
    k_sum = renew_period*k
    gamma_stock = gamma.ppf(PERCENT, a=k_sum, scale = theta)
    if(np.isnan(gamma_stock)):
        return 0
    else:
        action = np.max([int(gamma_stock)-inv, 0])
    return action
    
    
def get_end2end_iid(gbm, x, inv):
    feature = x[['vlt_mean_season', 'vlt_variance_season', 
               'vendor_vlt_mean', 'vendor_vlt_std', 'renew_period', 'initial_stock', 'mkt_prc', 'ave_3d_sale_near', 'ave_7d_sale_near',
               'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 'ave_3d_sale_far', 'ave_7d_sale_far', 'ave_14d_sale_far', 'ave_28d_sale_far',
               'ave_56d_sale_far', 'sales_14d_std']]
    feature['initial_stock'] = inv 
    feature = pd.DataFrame(feature, dtype='float').T
    action = np.max([gbm.predict(feature), 0])
    return int(action)

def get_end2end(gbm, x, inv, order_normal, order_gamma, order_eq, renew_period):
    '''feature = x[['vlt_mean_season', 'vlt_variance_season',  'vendor_vlt_mean', 'vendor_vlt_std', 'renew_period', 'initial_stock', 'mkt_prc', 'contract_stk_prc', 
                 'ave_3d_sale_near', 'ave_7d_sale_near','ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
                 'ave_3d_sale_far', 'ave_7d_sale_far','ave_14d_sale_far', 'ave_28d_sale_far', 'ave_56d_sale_far', 
                 'sales_14d_std']]'''

                 
    # feature = x[['vlt_mean_season', 'vlt_variance_season', 'vendor_vlt_mean', 'vendor_vlt_std','vendor_vlt_count', 
    #            'mkt_prc', 'contract_stk_prc', 'review_period', 'initial_stock',  
    #            'ave_3d_sale_near','ave_7d_sale_near', 'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
    #            'ave_90d_sale_near','ave_180d_sale_near', 'sales_180d_std',
    #            'sales_R', 'sales_VLT', 'sales_std', 'normal', 'gamma', 'eq']]
    
    feature = x[['vendor_vlt_count', 'vendor_vlt_mean', 'vendor_vlt_std',
              'vlt_mean_6mo', 'vlt_std_6mo', 'vlt_count_6mo', 'vlt_min_6mo',
               'vlt_max_6mo', 'vendor_vlt_min', 'vendor_vlt_max',
               'mkt_prc', 'contract_stk_prc', 'review_period', 'initial_stock',  
               'ave_3d_sale_near','ave_7d_sale_near', 'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
               'ave_90d_sale_near','ave_180d_sale_near', 'sales_180d_std', 
               'normal', 'gamma', 'eq']]


    feature['renew_period'] = renew_period
    feature['sales_R'] = feature['ave_180d_sale_near']*renew_period
    feature['sales_VLT'] = feature['ave_180d_sale_near']*feature['vendor_vlt_mean']
    feature['sales_std'] = np.sqrt(renew_period+feature['vendor_vlt_mean'])*feature['sales_180d_std']
    feature['normal'] = order_normal
    feature['gamma'] = order_gamma
    feature['eq'] = order_eq
    
    feature = pd.DataFrame(feature, dtype='float').T
    #action = np.max([gbm.predict(feature, categorical_feature=[0,1,2,3,4]), 0])
    action = np.max([gbm.predict(feature), 0])
    return int(action)

def get_normal_basestock(inv, sales_mean, sales_std, renew_period, VLT_mean, VLT_std, Z90 = 1.2816):
    VLT_std = 0
    action = np.max([0, sales_mean*(renew_period+VLT_mean)+
                     Z90*np.sqrt((renew_period+VLT_mean)*sales_std**2
                                 +sales_std**2*VLT_std)-inv])
    return int(action)

def get_normal_basestock2(record, inv, t, renew_period, VLT_mean, VLT_std, Z90 = 1.2816):
    sales = record[0:t]
    sales_mean = np.mean(sales)
    sales_std = np.std(sales)
    VLT_std = 0
    action = np.max([0, sales_mean*(renew_period+VLT_mean)+
                     Z90*np.sqrt((renew_period+VLT_mean)*sales_std**2
                                 +sales_std**2*VLT_std)-inv])
    return int(action)