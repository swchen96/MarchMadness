import pandas as pd
import numpy as np
from pprint import pprint as pprint
#
# data = pd.read_csv('team_stats_16-17.csv')
# teams = data[['Team name', 'Team ID']]
# pprint(teams)

data = pd.read_csv('team_stats_14-15.csv')

new_stats = pd.read_csv('rankings.csv').dropna()#.sort_values('Name').reset_index(drop = 'True')
#new_stats['Games'] = new_stats['W'] + new_stats['L']


new_stats = new_stats[['Name', 'Fouls', 'PFPG']]
new_stats.columns = ['Team name', 'Fouls', 'PFPG']
#pprint(new_stats)

full_data = data.merge(new_stats, on='Team name', how='left')#, on=)
pprint(full_data)

full_data.to_csv('team_stats_14-15.csv')
