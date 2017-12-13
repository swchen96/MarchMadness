import pandas as pd
import numpy as np
from pprint import pprint as pprint
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf




TRAIN_PATH = "TRAIN_DATA.csv"
TEST_PATH = "TEST_DATA.csv"

train_data = pd.read_csv(TRAIN_PATH)
train_X = train_data.ix[:, train_data.columns != 'Score 1'].values
train_Y = train_data['Score 1'].values

test_data = pd.read_csv(TEST_PATH)
test_Y = test_data['Score 1'].values
test_X = test_data.ix[:, test_data.columns != 'Score 1'].values




#runs lasso on the data set
# lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, \
#        copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
# lasso.fit(train_X, train_Y)
# lasso_coefs = lasso.coef_
# #print(lasso_coefs)
#
# #gets the nonzero coefficients from lasso
# non_zero_lasso = []
# for x in range(len(train_data.columns[:-1])):
#     if lasso_coefs[x] != 0:
#         non_zero_lasso.append(train_data.columns[x])
# #print(non_zero_lasso)
#
#
#
#
# lasso_prediction = lasso.predict(test_X)
# lasso_R2 = lasso.score(test_X, test_Y)
#
# print("lasso coefficients: ", lasso_coefs)
# print("lasso score: ", lasso_R2)
#
# plt.title('LASSO Score Prediction')
# plt.plot(test_Y, color='orange')
# plt.plot(lasso_prediction, color='#1F62A7')
# plt.xlabel('Games')
# plt.ylabel('Score')
# plt.show()


# test_X = test_X[non_zero_lasso].values

# #runs ridge regression on the nonzero coefficients from lasso
# clf = Ridge(alpha=4.0, copy_X=True, fit_intercept=True, max_iter=None,
#       normalize=False, random_state=None, solver='auto', tol=0.001)
# clf.fit(lasso_train_X,train_Y)
#
# ridge_prediction = clf.predict(test_X)
# ridge_R2 = clf.score(test_X, test_Y)
#
# print("ridge coefficients: ",clf.coef_)
# #print("prediction: ", ridge_prediction)
# print("ridge score: ", ridge_R2)
#
# plt.title('Ridge Regression Score Prediction')
# plt.plot(test_Y, color='orange')
# plt.plot(ridge_prediction, color='#1F62A7')
# plt.xlabel('Games')
# plt.ylabel('Score')
# plt.show()
#
#
# #runs linear regression on the nonzero coefficients from lasso
# LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# LR.fit(lasso_train_X, train_Y)
#
# linear_prediction = LR.predict(test_X)
# linear_R2 = LR.score(test_X, test_Y)
#
# print("linear coefficients: ", LR.coef_)
# print("linear score: ", linear_R2)
#
# plt.title('Linear Regression Score Prediction')
# plt.plot(test_Y, color='orange')
# plt.plot(linear_prediction, color='#1F62A7')
# plt.xlabel('Games')
# plt.ylabel('Score')
# plt.show()


# kmeans = KMeans((n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, \
#         precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm=’auto’))
# labels = kmeans.fit_predict(train_X, train_Y)

#exog = co2.ix[:, co2.columns != 'CO2']
