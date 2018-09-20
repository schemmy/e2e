#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:48:12 2018

@author: yyshi
"""

import numpy as np
import pandas as pd
import os
import warnings
import keras
import tensorflow as tf
from numpy import shape
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import random
import csv
from keras import optimizers
from sklearn import preprocessing

warnings.filterwarnings('ignore')


def readdata(url):
    
    # order table
    df1= pd.read_csv(url)
    
    df1 = df1.sample(frac=1).reset_index(drop=True)
    
    # df1 = df1[['int_org_num', 'item_third_cate_cd', 'brand_code', 'pur_bill_src_cd', 'create_day_of_week', 
    #            'vlt_mean_season', 'vlt_variance_season', 'vendor_vlt_mean', 'vendor_vlt_std','vendor_vlt_count', 
    #            'vendor_qtty_std', 'vendor_qtty_mean', 'mkt_prc', 'contract_stk_prc', 'review_period', 'initial_stock',  
    #            'ave_3d_sale_near','ave_7d_sale_near', 'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
    #            'ave_90d_sale_near','ave_180d_sale_near', 'sales_R', 'sales_VLT', 'sales_std','sales_180d_std', 
    #            'target_decision', 'actual_pur_qtty', 'normal', 'gamma', 'eq']]


    df1 = df1[['int_org_num', 'item_third_cate_cd', 'brand_code', 'pur_bill_src_cd', 'create_day_of_week', 
               'vendor_vlt_mean', 'vendor_vlt_std','vendor_vlt_count', 
               'vlt_mean_6mo', 'vlt_std_6mo', 'vlt_count_6mo', 'vlt_min_6mo',
               'vlt_max_6mo', 'vendor_vlt_min', 'vendor_vlt_max',
               'mkt_prc', 'contract_stk_prc', 'review_period', 'initial_stock',  
               'ave_3d_sale_near','ave_7d_sale_near', 'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
               'ave_90d_sale_near','ave_180d_sale_near','sales_180d_std', 
               'target_decision', 'actual_pur_qtty', 
               'normal', 'gamma', 'eq'
               ]]
            
    df_filtered = df1[(df1['ave_3d_sale_near'] == -1)]
    
    df1.drop(df_filtered.index, axis=0, inplace=True)
    
    # X = df1[['vendor_vlt_count', 'vendor_vlt_mean', 'vendor_vlt_std',
    #            'mkt_prc', 'contract_stk_prc', 'review_period', 'initial_stock',  
    #            'ave_3d_sale_near','ave_7d_sale_near', 'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
    #            'ave_90d_sale_near','ave_180d_sale_near', 'sales_180d_std', 
    #            'normal', 'gamma', 'eq']]

    X = df1[['vendor_vlt_count', 'vendor_vlt_mean', 'vendor_vlt_std',
              'vlt_mean_6mo', 'vlt_std_6mo', 'vlt_count_6mo', 'vlt_min_6mo',
               'vlt_max_6mo', 'vendor_vlt_min', 'vendor_vlt_max',
               'mkt_prc', 'contract_stk_prc', 'review_period', 'initial_stock',  
               'ave_3d_sale_near','ave_7d_sale_near', 'ave_14d_sale_near', 'ave_28d_sale_near', 'ave_56d_sale_near', 
               'ave_90d_sale_near','ave_180d_sale_near', 'sales_180d_std', 
               'normal', 'gamma', 'eq'
               ]]

    #'int_org_num', 'item_third_cate_cd', 'brand_code', 'pur_bill_src_cd', 'create_day_of_week', 
    #'sales_R', 'sales_VLT', 'sales_std',
    dataX = X
    
    dataY = df1[['target_decision']]

    datanum = len(dataX)
    
    X_train = dataX.loc[0:int(datanum*0.95)]
    y_train = dataY.loc[0:int(datanum*0.95)]
    
    X_test=dataX.loc[int(datanum*0.95):]
    y_test = dataY.loc[int(datanum*0.95):]
    
    ytest_bl1 = df1[['actual_pur_qtty']].loc[int(datanum*0.95):] #actual order
    ytest_bl2 = df1[['normal']].loc[int(datanum*0.95):] #normal basestock
    ytest_bl3 = df1[['gamma']].loc[int(datanum*0.95):] #gamma basestock
    ytest_bl4 = df1[['eq']].loc[int(datanum*0.95):] #Emperical quantile
    
    return df1, X_train, y_train, X_test, y_test, ytest_bl1, ytest_bl2, ytest_bl3, ytest_bl4