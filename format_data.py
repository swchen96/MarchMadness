'''
get all of the dataframes for all of the years of data
get all of the games and associated years
make 2 x vectors for each game and associate a score y for the first team in the vector
'''
import pandas as pd   
import os

team_data_root = "team_stats/team_stats_"
scores_root = "scores/"
id_map = pd.read_csv("id_map.csv")

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
	new = last + "-" + this + ".csv"
	return team_data_root + new

for year in games:
	df = pd.read_csv(make_data_path(year))
	for day in games[year]:
		for game in day:
			print(game)


# df17 = pd.read_csv(os.path.join(team_data_root, "16_17.csv"))
# df16 = pd.read_csv(os.path.join(team_data_root, "15_16.csv"))
# df15 = pd.read_csv(os.path.join(team_data_root, "14_15.csv"))
# df14 = pd.read_csv(os.path.join(team_data_root, "13_14.csv"))
# df13 = pd.read_csv(os.path.join(team_data_root, "12_13.csv"))
# df12 = pd.read_csv(os.path.join(team_data_root, "11_12.csv"))
# df11 = pd.read_csv(os.path.join(team_data_root, "10_11.csv"))
# df10 = pd.read_csv(os.path.join(team_data_root, "09_10.csv"))
# df09 = pd.read_csv(os.path.join(team_data_root, "08_09.csv"))
# df08 = pd.read_csv(os.path.join(team_data_root, "07_08.csv"))
# df07 = pd.read_csv(os.path.join(team_data_root, "06_07.csv"))
# df06 = pd.read_csv(os.path.join(team_data_root, "05_06.csv"))
# df05 = pd.read_csv(os.path.join(team_data_root, "04_05.csv"))
# df04 = pd.read_csv(os.path.join(team_data_root, "03_04.csv"))
# df03 = pd.read_csv(os.path.join(team_data_root, "02_03.csv"))

# year_to_df = {'2017': df17, 
# 			'2016': df16,
# 			'2015': df15, 
# 			'2014': df14,
# 			'2013': df13, 
# 			'2012': df12,
# 			'2011': df11, 
# 			'2010': df10,
# 			'2009': df09, 
# 			'2008': df08,
# 			'2007': df07, 
# 			'2006': df06,
# 			'2005': df05, 
# 			'2004': df04,
# 			'2003': df03}









