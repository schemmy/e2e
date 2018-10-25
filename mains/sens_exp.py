# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-19 16:26:07
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-24 17:48:52

import sys
sys.path.append('../')
from utils.benchmark import *
import pandas as pd
import argparse
from trainers.train_tc import *
import os, pickle
import warnings
from tqdm import tqdm
import math
import seaborn as sns
import scipy.stats as st
import datetime as dt
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str , default='v5',
                    help='v5: MLP; v6: RNN')
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--test_sku', type=str, default='None')
parser.add_argument('--b_value', type=int, default=None)
parser.add_argument('--model_to_load', type=str , default='e2e_v5_24.pkl',
                    help='model to be loaded for evaluation')
parser.add_argument('--train_check', type=str , default='None',
                    help='checkpoint for continuing training')
parser.add_argument('--model_path', type=str, default='../logs/torch/' ,
                    help='path for saving trained models')
parser.add_argument('--data_path', type=str, default='../data/1320_feature/',
                    help='path for data')
parser.add_argument('--log_path', type=str, default='../logs/torch_board/',
                    help='path for data')
parser.add_argument('--log_step', type=int, default=145,
                    help='step size for printing log info')
parser.add_argument('--save_step', type=int, default=24,
                    help='step size for saving trained models')
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
args = parser.parse_args()

hb = [5,7,9,12,15,20,25]
h = 1

o4 = pickle.load(open('../data/1320_feature/test_df.pkl', 'rb'))
o4['demand_RV_list_acm'] = o4['demand_RV_list'].apply(lambda x: np.cumsum(x))
numberOfRows = len(o4)

str_list = ['OPT', 
#             'E2E_NN', 
            'E2E_MLP', 
            # 'E2E_RNN', 
            'Normal', 'Gamma', 'Hist']

sens_cost = pd.DataFrame(index=hb, columns=str_list)
sens_hold = pd.DataFrame(index=hb, columns=str_list)
sens_sout = pd.DataFrame(index=hb, columns=str_list)


for (i,b) in enumerate(hb):

    args.b_value = b
    args.model_to_load = 'e2e_v5_24.pkl'
    model = End2End_v5_tc(device).to(device)
    trainer = Trainer(model, args, device)
    trainer.train_v5_tc()
    trainer.eval_v5_tc()
    pred_path = '../logs/torch/pred_v5.csv'
    pred_v5 = pd.read_csv(pred_path)

    o4['E2E_MLP_replen'] = (pred_v5['E2E_MLP_pred'] - o4['initial_stock']).clip(0)
    o4['E2E_MLP_inv_f'], o4['E2E_MLP_inv'] = zip(*o4.apply(get_inv, name='E2E_MLP_replen', axis=1))

    o4['demand_RV_%i' %b] = o4.apply(lambda x: sum(x['demand_RV_list']) if len(x['demand_RV_list']) <= b \
                    else sum(x['demand_RV_list'][:-int((x['review_period']+x['vlt_actual'])//(b+h))]), axis=1)

    o4['OPT_pred'] = o4['demand_RV_%i' %b]
    o4['OPT_replen'] = o4['OPT_pred'] - o4['initial_stock']
    o4['OPT_inv_f'], o4['OPT_inv'] = zip(*o4.apply(lambda x: get_inv(x, 'OPT_replen'), axis=1))

    Z = st.norm.ppf(1-(1-(b/(h+b)))/2)
    o4['Normal_pred'] = o4.apply(lambda x: int(x['mean_112']*1.1*(x['review_period']+x['vendor_vlt_mean'])
                                +Z*np.sqrt((x['review_period']+x['vendor_vlt_mean'])*x['std_140']**2
                                + x['std_140']**2 * x['vlt_std'])), axis=1)
    o4['Normal_replen'] = (o4['Normal_pred'] - o4['initial_stock']).clip(0)
    o4['Normal_inv_f'], o4['Normal_inv'] = zip(*o4.apply(get_inv, name='Normal_replen',  axis=1))

    o4['Gamma_pred'] = o4.apply(gamma_base, q=b/(h+b), axis=1)
    o4['Gamma_replen'] = (o4['Gamma_pred'] - o4['initial_stock']).clip(0)
    o4['Gamma_inv_f'],o4['Gamma_inv'] = zip(*o4.apply(get_inv, name='Gamma_replen', axis=1))

    o4['Hist_inv_f'], o4['Hist_inv'] = zip(*o4.apply(get_inv, name='actual_pur_qtty', axis=1))

    df_cost_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)
    df_holding_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)
    df_back_agg = pd.DataFrame(index=np.arange(0, numberOfRows), columns=str_list)

    for str1 in str_list:
        str2 = str1 + '_inv'
        df_holding_agg[str1] = o4[str2].apply(lambda x: h * sum([inv for inv in x if inv>0]) )
        df_back_agg[str1] = o4[str2].apply(lambda x: b * (0 if  x==[] else max(0, -x[-1])))
        df_cost_agg[str1] = df_holding_agg[str1] + df_back_agg[str1]
        
    sens_cost.iloc[i,:] = df_cost_agg[str_list].mean()
    sens_hold.iloc[i,:] = df_holding_agg[str_list].mean()
    sens_sout.iloc[i,:] = df_back_agg[str_list].mean()
    # sens_sout.loc[:,'OPT'] = sens_sout.loc[:,'OPT']/2


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sens_cost);
ax.legend(sens_cost.columns, loc=0)
ax.set_xlabel('b/h')
ax.set_ylabel('Total Cost')
ax.set_title('Sensitivity test on different ratios of b/h')
plt.savefig('../figures/eps/sens_cost.eps',dpi=200);

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sens_hold);
ax.legend(sens_hold.columns, loc=0)
ax.set_xlabel('b/h')
ax.set_ylabel('Total Cost')
ax.set_title('Sensitivity test on different ratios of b/h')
plt.savefig('../figures/eps/sens_hold.eps',dpi=200);

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sens_sout);
ax.legend(sens_sout.columns, loc=0)
ax.set_xlabel('b/h')
ax.set_ylabel('Stockout Cost')
ax.set_title('Sensitivity test on different ratios of b/h')
plt.savefig('../figures/eps/sens_sout.eps',dpi=200)

