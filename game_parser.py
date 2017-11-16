import urllib2
from bs4 import BeautifulSoup as bs
import pandas as pd

def get_scores(month, day, year):
	url = 'http://www.sports-reference.com/cbb/boxscores/index.cgi?month='+str(month)+'&day='+str(day)+'&year='+str(year)
	html = bs(urllib2.urlopen(url), 'lxml')
	#print html


	games = []

	for game_html in html.select('.game_summary.nohover'):
		data = []
		table = game_html.find('table', attrs={'class':'teams'})
		table_body = table.find('tbody')
		rows = table_body.find_all('tr')
		for row in rows:
		    cols = row.find_all('td')
		    cols = [ele.text.strip() for ele in cols]
		    data.append([ele for ele in cols if ele]) # Get rid of empty values
		games.append(data)

	#print games

	games_csv = []

	for game in games:
		if not len(game[1])==2:
			continue
		next_entry = []
		away = game[0]
		home = game[1]
		next_entry.append(away[0].encode('utf-8'))
		next_entry.append('@ '+home[0].encode('utf-8'))
		if int(away[1]) < int(home[1]):
			next_entry.append('L')
			next_entry.append(home[1].encode('utf-8')+'-'+away[1].encode('utf-8'))
		else:
			next_entry.append('W')
			next_entry.append(away[1].encode('utf-8')+'-'+home[1].encode('utf-8'))
		games_csv.append(next_entry)

	df = pd.DataFrame(data=games_csv)
	df.to_csv('parsed_gamelog.csv', index=False, index_label=False, header = ["Team", "Opponent", "Result", "Score"])


get_scores(11, 15, 2016)