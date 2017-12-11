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
  'Games',
	# 'PTS',
	'PPG',
	# 'OPP PTS', 
	'OPP PPG', 
	# 'SCR MAR', 
	'Win %', 
	'FTPG', 
	'FT%', 
	'FGPG', 
	'FG%',
	'3FGPG', 
	'3FG%',
	# 'Opp 3FG%', # we dont have it for many years
	# 'AST',
	'APG',
	# 'TO',
	'TOPG',
	# 'REB',
	'RPG',
	# 'OPP REB',
	'OPP RPG',
	# 'REB MAR',
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

remove_2016 = set(['Seattle', 'Old Dominion', 'Louisiana-Monroe', 'Central Michigan', 'Army', 'Grand Canyon'])
remove_2012 = set(['North Dakota', 'Ohio St.', 'Middle Tennessee', 'Coastal Carolina', 'South Florida', 'University of California',
                  'Lamar', 'Bowling Green St.', 'Quinnipiac', 'New Mexico St.', 'Alabama', 'Detroit'])
X = []
y = []
for year in games:
  print(year)
  df = pd.read_csv(make_data_path(year))
  for day in games[year]:
    for game_index in range(day.shape[0]):
      game = day.loc[game_index, :]
      if (year == 2012) and ((game['Team1'] in remove_2012) or (game['Team2'] in remove_2012)):
        continue
      if (year == 2016) and ((game['Team1'] in remove_2016) or (game['Team2'] in remove_2016)):
        continue
      team_id_1 = id_map.loc[id_map[0] == game['Team1'], 1].values[0]
      team_id_2 = id_map.loc[id_map[0] == game['Team2'], 1].values[0]
      team_1_stats = df.loc[df['Team ID'] == team_id_1][stats_we_want].values[0]
      if np.isnan(np.sum(team_1_stats)):
        print(year, game['Team1'])
      team_1_score = game['Score2']
      team_2_stats = df.loc[df['Team ID'] == team_id_2][stats_we_want].values[0]
      if np.isnan(np.sum(team_2_stats)):
      	print(year, game['Team2'])
      team_2_score = game['Score1']
      X.append(list(np.append(team_1_stats, team_2_stats)))
      y.append(team_1_score)
      X.append(list(np.append(team_2_stats, team_1_stats)))
      y.append(team_2_score)

with open('new_x.csv', 'w') as f:
	writer = csv.writer(f)
	for row in X:
		writer.writerow(row)

with open('new_y.csv', 'w') as f:
	writer = csv.writer(f)
	for row in y:
		writer.writerow([row])


'''
teams with nans:

2013 3FGA and 3P% errors: replaced with 0s
  2013 Marquette
  2013 Savannah St.
  2013 Loyola (MD)
  2013 St. John's (NY)
  2013 Vermont
  2013 Stephen F. Austin
  2013 Loyola (MD)
  2013 Oral Roberts
  2013 Bucknell
  2013 University of California
  2013 Marquette
  2013 Michigan St.
  2013 New Mexico St.
  2013 Green Bay
  2013 High Point
  2013 Charlotte
  2013 Purdue
  2013 Oral Roberts
  2013 Oral Roberts
  2013 Purdue
  2013 Loyola (MD)
  2013 St. John's (NY)
  2013 UCLA
  2013 Oklahoma
  2013 Marquette
  2013 University of California
  2013 Michigan St.
  2013 Marquette
  2013 Michigan St.

2016 AST and APG errors: replaced with 0s
  2016 Dayton
  2016 Middle Tennessee
  2016 Notre Dame
  2016 Texas
  2016 Coastal Carolina
  2016 Wagner
  2016 Louisiana-Lafayette
  2016 Gonzaga
  2016 Butler
  2016 Coastal Carolina
  2016 Notre Dame
  2016 Morehead St.
  2016 Morehead St.
  2016 Eastern Washington
  2016 North Carolina-Greensboro
  2016 Seattle
  2016 Notre Dame
  2016 Saint Mary's (CA)
  2016 Middle Tennessee
  2016 Coastal Carolina
  2016 Florida St.
  2016 Louisiana-Monroe
  2016 New Mexico St.
  2016 Saint Mary's (CA)
  2016 Morehead St.
  2016 Notre Dame
  2016 Gonzaga
  2016 Central Michigan
  2016 Eastern Washington
  2016 Fairfield
  2016 Texas A&M-Corpus Christi
  2016 Louisiana-Lafayette
  2016 North Carolina-Greensboro
  2016 Wagner
  2016 Western Carolina
  2016 Grand Canyon
  2016 Buffalo
  2016 Gonzaga
  2016 USC
  2016 Butler
  2016 Fresno St.
  2016 Florida St.
  2016 ETSU
  2016 Morehead St.
  2016 Old Dominion
  2016 ETSU
  2016 Old Dominion
  2016 Saint Mary's (CA)
  2016 Grand Canyon
  2016 Coastal Carolina
  2016 Louisiana-Lafayette
  2016 Morehead St.

2016 no FT: no solution yet
  2016 Old Dominion
  2016 Louisiana-Monroe

2016 no FG: no solution yet
  2016 Army

2016 no FG or FT: no solution yet
  2016 Seattle

2012 no 3FGA or 3FG%: replaced with 0s
  2012 Ohio St.
  2012 Middle Tennessee
  2012 Coastal Carolina
  2012 Middle Tennessee
  2012 Xavier
  2012 South Florida
  2012 South Florida
  2012 University of California
  2012 North Dakota
  2012 Lamar
  2012 Bowling Green St.
  2012 Quinnipiac
  2012 New Mexico St.
  2012 Ohio St.
  2012 Middle Tennessee
  2012 Ohio St.
  2012 Xavier
  2012 Ohio St.
  2012 Ohio St.
  2012 Alabama
  2012 Detroit
  2012 Virginia
  2012 Xavier
  2012 South Florida
'''

'''
still nans round 2: removed their 2012 games
2016 Seattle
2016 Louisiana-Monroe
2016 Central Michigan
2016 Army
2016 Old Dominion
2016 Grand Canyon

2012 North Dakota: removed their 2012 games
'''





