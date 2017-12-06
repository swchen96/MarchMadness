'''
get all of the dataframes for all of the years of data
get all of the games and associated years
make 2 x vectors for each game and associate a score y for the first team in the vector
'''
import pandas as pd   
import os
import numpy as np
import csv

stats_we_want = [
	# 'PTS',
	'PPG',
	# 'OPP PTS', 
	'OPP PPG', 
	'SCR MAR', 
	'Win %', 
	# 'FTA', #should we convert to per game?
	'FT%', 
	# 'FGA', #should we convert to per game?
	'FG%',
	# '3FGA', #should we convert to per game?
	'3FG%',
	'Opp 3FG%',
	# 'AST',
	'APG',
	# 'TO',
	'TOPG',
	# 'REB',
	'RPG',
	# 'OPP REB',
	'OPP RPG',
	'REB MAR',
	# 'DREB', #we don't have the data for most years
	# 'ST',
	'STPG',
	# 'Fouls',
	'PFPG']

team_data_root = "team_stats/team_stats_"
scores_root = "scores/"
id_map = pd.read_csv("id_map.csv", header = None)

games = {} #map year to list of dataframes
files = []

for filename in os.listdir(scores_root):
	if filename.endswith('.csv'):
		files.append(os.path.join(scores_root, filename))

for file in files:
	year = int(file[-8:-4])
	if year in games:
		games[year].append(pd.read_csv(file, encoding = 'ascii'))
	else:
		games[year] = [pd.read_csv(file, encoding = 'ascii')]

def make_data_path(year):
	this = str(year)[2:]
	last = str(int(this) - 1)
	if len(last) == 1:
		last = "0" + last
	new = last + "_" + this + ".csv"
	return team_data_root + new

X = []
y = []
for year in games:
	print(year)
	df = pd.read_csv(make_data_path(year))
	for day in games[year]:
		for game_index in range(day.shape[0]):
			game = day.loc[game_index, :]
			team_id_1 = id_map.loc[id_map[0] == game['Team1'], 1].values[0]
			team_id_2 = id_map.loc[id_map[0] == game['Team2'], 1].values[0]
			team_1_stats = df.loc[df['Team ID'] == team_id_1][stats_we_want].values[0]
			team_1_score = game['Score2']
			team_2_stats = df.loc[df['Team ID'] == team_id_2][stats_we_want].values[0]
			team_2_score = game['Score1']
			X.append(list(np.append(team_1_stats, team_2_stats)))
			y.append(team_1_score)
			X.append(list(np.append(team_2_stats, team_1_stats)))
			y.append(team_2_score)

with open('dont overwrite accidentally', 'w') as f:
	writer = csv.writer(f)
	for row in X:
		writer.writerow(row)

with open('dont overwrite accidentally', 'w') as f:
	writer = csv.writer(f)
	for row in y:
		writer.writerow([row])










