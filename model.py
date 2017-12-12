import pandas as pd
import numpy as np
from pprint import pprint as pprint
from sklearn.linear_model import Ridge, Lasso

TRAIN_PATH = "TRAIN_DATA.csv"
TEST_PATH = "TEST_DATA.csv"

data = pd.read_csv(TRAIN_PATH)
X = data.ix[:, data.columns != 'Score 1'].values
Y = data['Score 1'].values

#runs lasso on the data set
lasso = Lasso(alpha=1, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
lasso.fit(X,Y)
lasso_coefs = lasso.coef_
#print(lasso_coefs)

#gets the nonzero coefficients from lasso
non_zero_lasso = []
nonzero_cols = []
for x in range(len(data.columns[:-1])):
    if lasso_coefs[x] != 0:
    	nonzero_cols.append(x)
        non_zero_lasso.append(data.columns[x])
#print(non_zero_lasso)

ridgeX = data.ix[:, nonzero_cols].values
ridgeY = Y

#runs ridge regression on the dataset
clf = Ridge(alpha=1, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
clf.fit(ridgeX, ridgeY)

#print("Ridge coefficients")
#print(clf.coef_)

test_data = pd.read_csv(TEST_PATH)
testX = test_data.ix[:, nonzero_cols].values
#print testX.shape
testY = test_data['Score 1'].values
print clf.score(testX,testY)