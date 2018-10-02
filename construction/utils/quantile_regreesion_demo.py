# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-09-27 10:22:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-09-27 10:25:38
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(1)

# Use sklearn or lightgbm?
USE_SKLEARN = False # Toggle this to observe issue.


# Quantile to Estimate
alpha = 0.9
# Training data size
N_DATA = 1000
# Function to Estimate
def f(x):
    """The function to predict."""
    return x * np.sin(x)

# model parameters
LEARNING_RATE = 0.1
N_ESTIMATORS = 100
MAX_DEPTH = -1
NUM_LEAVES = 31 # lgbm only
OBJECTIVE = 'quantile' # lgbm only, 'quantile' or 'quantile_l2'
REG_SQRT = True # lgbm only
if USE_SKLEARN:
    if MAX_DEPTH < 0:  # sklearn grows differently than lgbm.
        print('Max Depth specified is incompatible with sklearn. Changing to 3.')
        MAX_DEPTH = 3

#---------------------- DATA GENERATION ------------------- #

#  First the noiseless case
X = np.atleast_2d(np.random.uniform(0, 10.0, size=N_DATA)).T
X = X.astype(np.float32)

# Observations
y = f(X).ravel()

dy = 1.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise
y = y.astype(np.float32)

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
xx = np.atleast_2d(np.linspace(0, 10, 9999)).T
xx = xx.astype(np.float32)


# Train high, low, and mean regressors.
# ------------------- HIGH/UPPER BOUND ------------------- #
if USE_SKLEARN:
    clfh = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE, min_samples_leaf=9,
                min_samples_split=9)

    clfh.fit(X, y)
else:
    ## ADDED
    clfh = lgb.LGBMRegressor(objective = OBJECTIVE,
                            alpha = alpha,
                            num_leaves = NUM_LEAVES,
                            learning_rate = LEARNING_RATE,
                            n_estimators = N_ESTIMATORS,
                            reg_sqrt = REG_SQRT,
                            max_depth = MAX_DEPTH)
    clfh.fit(X, y,
            #eval_set=[(X, y)],
            #eval_metric='quantile'
           )
    ## END ADDED

# ------------------- LOW/LOWER BOUND ------------------- #

if USE_SKLEARN:
    clfl = GradientBoostingRegressor(loss='quantile', alpha=1.0-alpha,
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, min_samples_leaf=9,
        min_samples_split=9)

    clfl.fit(X, y)
else:
    ## ADDED
    clfl = lgb.LGBMRegressor(objective = OBJECTIVE,
                            alpha = 1.0 - alpha,
                            num_leaves = NUM_LEAVES,
                            learning_rate = LEARNING_RATE,
                            n_estimators = N_ESTIMATORS,
                            reg_sqrt = REG_SQRT,
                            max_depth = MAX_DEPTH)
    clfl.fit(X, y,
            #eval_set=[(X, y)],
            #eval_metric='quantile'
            )
    ## END ADDED

# ------------------- MEAN/PREDICTION ------------------- #

if USE_SKLEARN:
    clf = GradientBoostingRegressor(loss='ls',
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, min_samples_leaf=9,
            min_samples_split=9)
    clf.fit(X, y)
else:
    ## ADDED
    clf = lgb.LGBMRegressor(objective = 'regression',
                            num_leaves = NUM_LEAVES,
                            learning_rate = LEARNING_RATE,
                            n_estimators = N_ESTIMATORS,
                            max_depth = MAX_DEPTH)

    clf.fit(X, y,
            #eval_set=[(X, y)],
            #eval_metric='l2',
            #early_stopping_rounds=5
            )
    ## END ADDED

# ---------------- PREDICTING ----------------- #

# Make the prediction on the meshed x-axis
y_pred = clf.predict(xx)
y_lower = clfl.predict(xx)
y_upper = clfh.predict(xx)

# Check calibration by predicting the training data.
y_autopred = clf.predict(X)
y_autolow = clfl.predict(X)
y_autohigh = clfh.predict(X)
frac_below_upper = round(np.count_nonzero(y_autohigh > y) / len(y),3)
frac_above_upper = round(np.count_nonzero(y_autohigh < y) / len(y),3)
frac_above_lower = round(np.count_nonzero(y_autolow < y) / len(y),3)
frac_below_lower = round(np.count_nonzero(y_autolow > y) / len(y),3)

# Print calibration test
print('fraction below upper estimate: \t actual: ' + str(frac_below_upper) + '\t ideal: ' + str(alpha))
print('fraction above lower estimate: \t actual: ' + str(frac_above_lower) + '\t ideal: ' + str(alpha))

# ------------------- PLOTTING ----------------- #

plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=3, label=u'Observations')
plt.plot(xx, y_pred, 'r-', label=u'Mean Prediction')
plt.plot(xx, y_upper, 'k-')
plt.plot(xx, y_lower, 'k-')
plt.fill(np.concatenate([xx, xx[::-1]]),
            np.concatenate([y_upper, y_lower[::-1]]),
            alpha=.5, fc='b', ec='None', label=(str(round(100*(alpha-0.5)*2))+'% prediction interval'))
plt.scatter(x=X[y_autohigh < y], y=y[y_autohigh < y], s=20, marker='x', c = 'red', 
        label = str(round(100*frac_above_upper,1))+'% of training data above upper (expect '+str(round(100*(1-alpha),1))+'%)')
plt.scatter(x=X[y_autolow > y], y=y[y_autolow > y], s=20, marker='x', c = 'orange', 
        label = str(round(100*frac_below_lower,1))+ '% of training data below lower (expect '+str(round(100*(1-alpha),1))+'%)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.title(  '  Alpha: '+str(alpha) +
            '  Sklearn?: '+str(USE_SKLEARN) +
            '  N_est: '+str(N_ESTIMATORS) +
            '  L_rate: '+str(LEARNING_RATE) +
            '  N_Leaf: '+str(NUM_LEAVES) + 
            '  Obj: '+str(OBJECTIVE) +
            '  R_sqrt: '+str(int(REG_SQRT))
        )

plt.show()