import urllib2
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import date, timedelta as td

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
		away_name = away[0]
		if away[0][-1] == ')' and away[0][-2].isdigit():
			t = away[0].find('(')
			away_name = away[0][:t-1]
		home_name = home[0]
		if home[0][-1] == ')' and home[0][-2].isdigit():
			t = home[0].find('(')
			home_name = home[0][:t-1]

		next_entry.append(away_name.encode('utf-8'))
		next_entry.append(home_name.encode('utf-8'))
		if int(away[1]) < int(home[1]):
			next_entry.append('L')
			next_entry.append(home[1].encode('utf-8')+'-'+away[1].encode('utf-8'))
		else:
			next_entry.append('W')
			next_entry.append(away[1].encode('utf-8')+'-'+home[1].encode('utf-8'))
		games_csv.append(next_entry)

	df = pd.DataFrame(data=games_csv)
	if not df.empty:
		df.to_csv('scores/log'+str(month)+'-'+str(day)+'-'+str(year)+'.csv', index=False, index_label=False, header = ["Team", "Opponent", "Result", "Score"])

tourney_dates = {}
tourney_dates['2017'] = [date(2017, 3, 14), date(2017, 4, 3)]
tourney_dates['2016'] = [date(2016, 3, 15), date(2016, 4, 4)]
tourney_dates['2015'] = [date(2015, 3, 17), date(2015, 4, 6)]
tourney_dates['2014'] = [date(2014, 3, 18), date(2014, 4, 7)]
tourney_dates['2013'] = [date(2013, 3, 19), date(2013, 4, 8)]
tourney_dates['2012'] = [date(2012, 3, 13), date(2012, 4, 2)]
tourney_dates['2011'] = [date(2011, 3, 15), date(2011, 4, 4)]
tourney_dates['2010'] = [date(2010, 3, 16), date(2010, 4, 5)]
tourney_dates['2009'] = [date(2009, 3, 17), date(2009, 4, 6)]
tourney_dates['2008'] = [date(2008, 3, 18), date(2008, 4, 7)]
tourney_dates['2007'] = [date(2007, 3, 13), date(2007, 4, 2)]
tourney_dates['2006'] = [date(2006, 3, 14), date(2006, 4, 3)]
tourney_dates['2005'] = [date(2005, 3, 15), date(2005, 4, 4)]
tourney_dates['2004'] = [date(2004, 3, 16), date(2004, 4, 5)]


for year in tourney_dates.keys():
	d0 = tourney_dates[year][0]
	d1 = tourney_dates[year][1]
	delta = d1-d0
	for i in range(delta.days+1):
		cur = d0+td(days=i)
		get_scores(cur.month, cur.day, cur.year)
