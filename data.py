import pandas as pd
import numpy as np
from pprint import pprint as pprint

data = pd.read_csv('team_stats.csv')
#pprint(data)

new_stats = pd.read_csv('rankings.csv').dropna()
#pprint(new_stats)


new_stats = new_stats[['Name', '3FGA']]
new_stats.columns = ['Team name', '3FGA']
#pprint(new_stats)

full_data = data.merge(new_stats, how='left')#, on=)
#pprint(full_data)

full_data.to_csv('team_stats.csv')
