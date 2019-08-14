import pandas as pd
import math
import csv
import random
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

base_elo = 1600
team_elos = {}
team_stats = {}
x = []
y = []

def initialize_data(M_data, O_data, T_data): # To merge T,O,M into a single dataframe

    # To remove the certain cols
    new_Mdata = M_data.drop(['Rk','Arena'],axis=1)
    new_Odata = O_data.drop(['Rk','G'],axis=1)
    new_Tdata = T_data.drop(['Rk','G'],axis=1)

    # Merge the 3 csv charts
    team_stats1 = pd.merge(new_Mdata, new_Odata, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tdata, how='left', on='Team')
    
    # Choose shared column 'Team' as index(row key) 
    return team_stats1.set_index('Team', inplace=False, drop=True)

def get_elo(team):
    try:
        return team_elos[team]
    except:
        # If there's no available elo score, assign the base_elo to it.
        team_elos[team] = base_elo
        return team_elos[team]

def calc_elo(win_team, lose_team):
    winner_elo = get_elo(win_team)
    loser_elo = get_elo(lose_team)

    elo_diff = winner_elo - loser_elo
    exp = (elo_diff  * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # 根据elo级别修改K值
    if winner_elo < 2100:
        k = 32
    elif winner_elo >= 2100 and winner_elo < 2400:
        k = 24
    else:
        k = 16

    # 更新 elo 数值
    new_winner_elo = round(winner_elo + (k * (1 - odds)))      
    new_loser_elo = round(loser_elo + (k * (0 - odds)))
    return new_winner_elo, new_loser_elo

def build_dataSet(all_data):
    print("Building data set..")
    X = []
    for index, row in all_data.iterrows():

        Wteam = row['win_T']
        Lteam = row['lose_T']

        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        # Add 100 elo score to home team
        if row['win_L'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        # use elo score as the first feature
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # add all data categories
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)

        # Randomly assign win team to one side. 0 in y means team1 wins, vice versa.
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        # update new elo score
        new_winner_elo, new_loser_elo = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_elo
        team_elos[Lteam] = new_loser_elo

    return np.nan_to_num(X), y

if __name__ == '__main__':
    
    M_data = pd.read_csv('17-18_M.csv')
    O_data = pd.read_csv('17-18_O.csv')
    T_data = pd.read_csv('17-18_T.csv')

    team_stats = initialize_data(M_data, O_data, T_data)
    result_data = pd.read_csv('17-18_wl.csv')
    X, y = build_dataSet(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression(solver='liblinear')
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    print("cross-validation accuracy:" + str(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean()))

def predict_winner(team_1, team_2, model):
    features = []

    # team 1，客场队伍
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # team 2，主场队伍
    # features.append(get_elo(team_2) + 100)
    # for key, value in team_stats.loc[team_2].iteritems():
    #     features.append(value)

    features = np.nan_to_num(features)
    print(features)
    return model.predict_proba([features])

print('Predicting on new schedule..')
schedule1819 = pd.read_csv('18-19_wl.csv')
result = []
for index, row in schedule1819.iterrows():

    if row['win_L'] == 'H':
        team1 = row['win_T']
        team2 = row['lose_T']
    else:
        team1 = row['lose_T']
        team2 = row['win_T']
        
    pred = predict_winner(team1, team2, model)
    print(pred)
    prob = pred[0][0]
    if prob > 0.5:
        winner = team1
        loser = team2
        result.append([winner, loser, prob])
    else:
        winner = team2
        loser = team1
        result.append([winner, loser, 1 - prob])

with open('18-19predict.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['win', 'lose', 'probability'])
    writer.writerows(result)
    print('done.')

