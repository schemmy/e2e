# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:30:40
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-19 16:04:41


import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm
import math
import seaborn as sns

from scipy.stats import gamma
import datetime as dt

import matplotlib.pyplot as plt


df_compare.plot.bar(rot=0)
plt.savefig('../figures/eps/exp2.eps', dpi=200)


sl_qtl = df_cost.sort_values('Ave_sales').groupby(df_cost.index//360).mean()
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(np.log(sl_qtl.iloc[:,:-2].astype(float)).replace(-np.inf, 0));
ax.legend(sl_qtl.columns[:-2], loc=0)
ax.set_xlabel('$i$-th quantile of averaged demand');
ax.set_ylabel('total cost(log)');
ax.set_ylim((0,10));
plt.savefig('../figures/eps/qtl1.eps', dpi=200)