import pandas as pd
import numpy as np
from pprint import pprint as pprint
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

tournament = pd.read_csv('16-17_tournament.csv')
#pprint(tournament)
round64 = set(tournament['round64'].values)
round32 = set(tournament['round32'].values)
round16 = set(tournament['round16'].values)
round8 = set(tournament['round8'].values)
round4 = set(tournament['round4'].values)
round2 = set(tournament['round2'].values)
winner = set(tournament['winner'].values)

teams = [round32, round16, round8, round4, round2, winner]


filename = "ridge_lasso_brackets.txt"
predictions = open(filename).read().splitlines()
# lines = lines.read()
# lines = lines.splitlines()


scores = []

best = None
best_score = 0

worst = None
worst_score = 2000

avg_score = 0

score_round = [10,20,40,80,160,320]

buckets = [0 for x in range(20)]

for prediction in predictions:
    prediction = ast.literal_eval(prediction)
    score = 0
    for x in range(len(prediction)):
        predicted_round = set(prediction[x])
        correct = predicted_round.intersection(teams[x])
        #print(x,correct)
        score += len(correct)*score_round[x]
    if score > best_score:
        best = prediction
        best_score = score
    if score < worst_score:
        worst = prediction
        worst_score = score

    avg_score += score
    scores.append(score)

#histogram = np.histogram(scores, bins=20, range)
avg_score = avg_score/(len(scores))

print("worst: ", worst)
print("worst score: ", worst_score)

print("best: ", best)
print("best score: ", best_score)

print("avg score: ", avg_score)

fig, ax = plt.subplots()
plt.title('Predicted Score Average: LASSO and Ridge')
plt.hist(scores, bins='auto')
plt.xlabel('score')
plt.ylabel('Number of brackets in the range')
plt.axvline(np.array(scores).mean(), color='r', linestyle='dashed', linewidth=2)
plt.axvline(np.array([715.40]).mean(), color='black', linestyle='dashed', linewidth = 2)
plt.axvline(np.array([best_score]).mean(), color='green', linestyle='dashed', linewidth = 2)
high = mpatches.Patch(color='green', label='Best prediction: ' + str(best_score))
avg = mpatches.Patch(color='red', label='Average Score: ' + str(avg_score))
espn = mpatches.Patch(color='black', label='ESPN Average: ' + str(715))
plt.legend(handles=[avg,espn, high])

plt.show()
