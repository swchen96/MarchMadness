import pandas as pd
import numpy as np
from numpy.random import normal


VARIANCE = 135.037908716

def get_training_variance():
	'''
	output:
		variance -> float (variance of the scores in the training data)
	'''
	df = pd.read_csv('TRAIN_DATA.csv')
	scores = df['Score 1'].values
	return np.var(scores)

def pick_winner(team1_score, team2_score):
	'''
	input:
		team1_score -> int
		team2_score -> int

	output:
		winner -> int (1 or 2)
	'''
	predict_1 = normal(team1_score, VARIANCE ** .5)
	predict_2 = normal(team2_score, VARIANCE ** .5)
	if predict_1 > predict_2:
		return 1
	else:
		return 2



# VARIANCE = get_training_variance()
