import pandas as pd
import numpy as np
import random
from pprint import pprint as pprint
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm


TRAIN_PATH = "TRAIN_DATA.csv"
TEST_PATH = "TEST_DATA.csv"

train_data = pd.read_csv(TRAIN_PATH)
train_X = train_data.ix[:, train_data.columns != 'Score 1'].values
train_Y = train_data['Score 1'].values

test_data = pd.read_csv(TEST_PATH)
test_Y = test_data['Score 1'].values


#runs lasso on the data set
lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
lasso.fit(train_X, train_Y)
lasso_coefs = lasso.coef_
#print(lasso_coefs)

#gets the nonzero coefficients from lasso
non_zero_lasso = []
nonzero_cols = []
for x in range(len(train_data.columns[:-1])):
    if lasso_coefs[x] != 0:
        non_zero_lasso.append(train_data.columns[x])
        nonzero_cols.append(x)
#print(non_zero_lasso)


test_X = test_data.ix[:, test_data.columns != 'Score 1'].values
#test_X = test_X[non_zero_lasso].values

'''
lasso_prediction = lasso.predict(test_X)
lasso_R2 = lasso.score(test_X, test_Y)

print("lasso coefficients: ", lasso_coefs)
print("lasso score: ", lasso_R2)

plt.title('LASSO Score Prediction')
plt.plot(test_Y, color='orange')
plt.plot(lasso_prediction, color='#1F62A7')
plt.xlabel('Games')
plt.ylabel('Score')
plt.show()
'''

lasso_train_X = train_data.ix[:, nonzero_cols].values
lasso_test_X = test_data.ix[:, nonzero_cols].values

#runs ridge regression on the nonzero coefficients from lasso


ridge_lasso = Ridge(alpha=4.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
ridge_lasso.fit(lasso_train_X,train_Y)

ridge_prediction = ridge_lasso.predict(lasso_test_X)
ridge_R2 = ridge_lasso.score(lasso_test_X, test_Y)

'''
print("ridge coefficients: ",ridge_lasso.coef_)
#print("prediction: ", ridge_prediction)
print("ridge score: ", ridge_R2)
#
plt.title('Ridge Regression Score Prediction')
plt.plot(test_Y, color='orange')
plt.plot(ridge_prediction, color='#1F62A7')
plt.xlabel('Games')
plt.ylabel('Score')
plt.show()
'''



stats2016 = pd.read_csv("main_64_stats_16_17.csv")

#returns np matrix with two rows, one for each team's X
def get_game_row(team1name, team2name):
	first = stats2016[stats2016['Team name'] == team1name]
	second = stats2016[stats2016['Team name'] == team2name]
	first = first.values[0, 2:]
	second = second.values[0, 2:]
	return np.array([np.append(first,second), np.append(second,first)])


def score_to_prob(diff):
	return norm.cdf(diff/5)

#tournament code
cur_round = []
tournament = []
for row in stats2016.values:
	tournament.append(row[0])

while len(tournament) > 1:
	cur_round = []
	for i in range(len(tournament)/2):
		team1 = tournament.pop(0)
		team2 = tournament.pop(0)
		cur = get_game_row(team1, team2)
		print "outcome for "+team1+" vs "+team2
		pred = ridge_lasso.predict(cur[:, nonzero_cols])
		#pred = lasso.predict(cur)
		winner = team1 if pred[0] > pred[1] else team2
		#print "winner: "+winner
		#print pred
		if random.random() < score_to_prob(pred[0] - pred[1]):
			cur_round.append(team1)
			print team1
		else:
			cur_round.append(team2)
			print team2

	tournament = list(cur_round)




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


#exog = co2.ix[:, co2.columns != 'CO2']