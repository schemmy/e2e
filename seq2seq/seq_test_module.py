#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:31:00 2018

@author: yyshi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms import get_opt, get_EQ, get_end2end_iid, get_end2end, get_normal_basestock, get_gamma_basestock, get_normal_basestock2
import lightgbm as lgb

def countOccurrences(arr, x):
    res = 0
    for i in range(len(arr)):
        if x == arr[i]:
            res += 1
    return res
            
def seqtest(sku, dc, START_DAY, inv0, df_opt, gbm_dq, end_of_horizon, arrive_time, order_time, demand, feature, OPTION):
    inv = inv0
    h = 1
    b = 9
    history_inv = []
    order_list = []
    h_cost_list = []
    b_cost_list = []
    cost_list = []
    #print('Initial Inventory:', inv0)
    for t in range(order_time[0], end_of_horizon):
        if t in arrive_time:
            occurance = countOccurrences(arrive_time, t)
            while(occurance>0):
                i = arrive_time.index(t)+occurance-1
                #print('receiving ',i , 'th order at time', t, 'Order qtty:', order_list[i])
                inv = inv + order_list[i]
                occurance = occurance-1
        
        if t in order_time:
            occurance = countOccurrences(order_time, t)
            count = occurance
            while(occurance>0):
                i = order_time.index(t)+ count - occurance
                cur_state = feature.iloc[i]
                #renew_period = cur_state['review_period']
                renew_period = cur_state['review_period']
                if(i<len(order_time)-1):
                    renew_period = order_time[i+1]-order_time[i]
                else:
                    renew_period = np.max([76-order_time[i]-int(cur_state['vlt_mean_season']), 0]) #50
                '''---------------------- OPTION 0: optimal DP ---------------------------------------'''
                '''---------------------- OPTION 1: Emperical Quantile -------------------------------'''
                '''---------------------- OPTION 2: End2end DP ---------------------------------------'''
                '''---------------------- OPTION 3: Normal Basestock ------------------------------------'''
                '''---------------------- OPTION 4: Gamma Basestock ---------------------------------'''
                if(OPTION==0): 
                    order = get_opt(df_opt, inv, i, sku, dc)
                elif(OPTION==1): 
                    order = get_EQ(demand, inv, t+START_DAY, renew_period+int(cur_state['vlt_mean_season']), PERCENT=90)   
                elif(OPTION==2): #end2end DP
                    order_normal = get_normal_basestock2(demand, inv, t+START_DAY, renew_period, cur_state['vlt_mean_season'], cur_state['vlt_variance_season'], Z90 = 1.2816)
                    order_gamma = get_gamma_basestock(demand, inv, t+START_DAY, renew_period+int(cur_state['vlt_mean_season']), PERCENT=0.9)
                    order_eq = get_EQ(demand, inv, t+START_DAY, renew_period+int(cur_state['vlt_mean_season']), PERCENT=90)   
                    order = get_end2end(gbm_dq, cur_state, inv, order_normal, order_gamma, order_eq, renew_period)
                elif(OPTION==3): #normal basestock
                    #order = get_normal_basestock(inv, cur_state['ave_180d_sale_near'], cur_state['sales_180d_std'], renew_period, cur_state['vendor_vlt_mean'], cur_state['vlt_variance_season'], Z90 = 1.2816)
                    order = get_normal_basestock2(demand, inv, t+START_DAY, renew_period, cur_state['vlt_mean_season'], cur_state['vlt_variance_season'], Z90 = 1.2816)
                elif(OPTION==4): #gamma base stock
                    order = get_gamma_basestock(demand, inv, t+START_DAY, renew_period+int(cur_state['vlt_mean_season']), PERCENT=0.9)
                elif(OPTION==5): #historical decision
                    order = cur_state['actual_pur_qtty']
                else: 
                    order = 0
                order_list.append(order)
                occurance = occurance-1
            
            #print('placing ', i, 'th order at time', t, 'Order qtty:', order, 'order list:', order_list)

        inv = inv - demand[START_DAY+t]
     
        h_cost = h*np.maximum(0, inv)
        b_cost = b*np.maximum(0, -inv)
        h_cost_list.append(h_cost)
        b_cost_list.append(b_cost)
        cost_list.append(h_cost+b_cost)
        
        history_inv.append(int(inv))
    return history_inv, sum(cost_list), sum(b_cost_list), sum(h_cost_list)