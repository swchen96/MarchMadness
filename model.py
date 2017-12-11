import pandas as pd
import numpy as np
from pprint import pprint as pprint
from sklearn.linear_model import Ridge, Lasso

TRAIN_PATH = "FULL_DATA.csv"
TEST_PATH = "FULL_DATA.csv"

data = pd.read_csv(TRAIN_PATH)
X = data.ix[:, data.columns != 'Score 1'].values
Y = data['Score 1'].values

#runs lasso on the data set
lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
lasso.fit(X,Y)
lasso_coefs = lasso.coef_
print(lasso_coefs)

#gets the nonzero coefficients from lasso
non_zero_lasso = []
for x in range(len(data.columns[:-1])):
    if lasso_coefs[x] != 0:
        non_zero_lasso.append(data.columns[x])
print(non_zero_lasso)


#runs ridge regression on the dataset
# clf = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#       normalize=False, random_state=None, solver='auto', tol=0.001)
# clf.fit(X,Y)


#print(clf.coef_)
#print(clf.predict(X))
#print(clf.score(X,Y))

#exog = co2.ix[:, co2.columns != 'CO2']
