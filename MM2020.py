import os
import pandas as pd
import numpy as np
import xgboost as xgb
import math

os.chdir('C:\\Users\\jogel\\Desktop\\MM2020')

# read csv files
raw_teams = pd.read_csv('MTeams.csv', sep = ',') # get FirstD1Season from here
raw_seeds = pd.read_csv('MNCAATourneySeeds.csv', sep = ',') # get Seed from here in format W01

raw_reg = pd.read_csv('MRegularSeasonDetailedResults.csv', sep =  ',')
raw_tour = pd.read_csv('MNCAATourneyDetailedResults.csv', sep = ',')

raw_conf_tour = pd.read_csv('MConferenceTourneyGames.csv', sep = ',')

subm = pd.read_csv('MSampleSubmissionStage1_2020.csv', sep = ',')

d1_dict = {}
for index, row in raw_teams.iterrows():
    d1_dict[row['TeamID']] = row['FirstD1Season']

stage1_train = range(2003, 2015)
stage1_test = range(2015, 2020)

train = pd.DataFrame(columns = ['ID', 'Delta', 'SeedDiff',
                                '1FGM', '1FGA', '1FGM3', '1FGA3', '1FTM', '1FTA',
                                '1OR', '1DR', '1AST', '1TO', '1STL', '1BLK', '1PF',
                                '1Score', '1WRatio',
                                '2FGM', '2FGA', '2FGM3', '2FGA3', '2FTM', '2FTA',
                                '2OR', '2DR', '2AST', '2TO', '2STL', '2BLK', '2PF',
                                '2Score', '2WRatio'])

# create training sets on 2003-2014
for s in stage1_train:
    print(s)
    # regular and tournament games detailed
    reg = raw_reg[raw_reg['Season'] == s]
    tour = raw_tour[raw_tour['Season'] == s]
    conf_tour = raw_conf_tour[raw_conf_tour['Season'] == s]
    
    # conference tourney list
    conf_games = []
    
    for index, row in conf_tour.iterrows():
        game = str(row['DayNum']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])
        conf_games.append(game)
    
    # seeds dictionary for upset variable
    # also, list of unique team IDs in given season's bracket
    seeds = raw_seeds[raw_seeds['Season'] == s]
    seeds_dict = {}
    
    for index, row in seeds.iterrows():
        seeds_dict[row['TeamID']] = int(row['Seed'][1:3])
    
    # reg season stats
    reg_rec = {}

    for t in seeds_dict.keys():
        #['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'AST', 'TO', 'STL', 'BLK', 'PF', 'Delta']
        reg_rec[t] = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]

        won = 0
        denom = 0

        for index, row in reg.iterrows():
            game = str(row['DayNum']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])
            if (row['WTeamID'] == t) and ((row['LTeamID'] in seeds_dict.keys()) | (game in conf_games)):
                reg_rec[t][0].append(row['WFGM'])
                reg_rec[t][1].append(row['WFGA'])
                reg_rec[t][2].append(row['WFGM3'])
                reg_rec[t][3].append(row['WFGA3'])
                reg_rec[t][4].append(row['WFTM'])
                reg_rec[t][5].append(row['WFTA'])
                reg_rec[t][6].append(row['WOR'])
                reg_rec[t][7].append(row['WDR'])
                reg_rec[t][8].append(row['WAst'])
                reg_rec[t][9].append(row['WTO'])
                reg_rec[t][10].append(row['WStl'])
                reg_rec[t][11].append(row['WBlk'])
                reg_rec[t][12].append(row['WPF'])
                reg_rec[t][13].append(row['WScore'] - row['LScore'])
                won += 1
                denom += 1
                
            elif (row['LTeamID'] == t) and ((row['LTeamID'] in seeds_dict.keys()) | (game in conf_games)):
                reg_rec[t][0].append(row['LFGM'])
                reg_rec[t][1].append(row['LFGA'])
                reg_rec[t][2].append(row['LFGM3'])
                reg_rec[t][3].append(row['LFGA3'])
                reg_rec[t][4].append(row['LFTM'])
                reg_rec[t][5].append(row['LFTA'])
                reg_rec[t][6].append(row['LOR'])
                reg_rec[t][7].append(row['LDR'])
                reg_rec[t][8].append(row['LAst'])
                reg_rec[t][9].append(row['LTO'])
                reg_rec[t][10].append(row['LStl'])
                reg_rec[t][11].append(row['LBlk'])
                reg_rec[t][12].append(row['LPF'])
                reg_rec[t][13].append(row['LScore'] - row['WScore'])
                denom += 1

        reg_rec[t][0] = round(np.mean(reg_rec[t][0]), 5)
        reg_rec[t][1] = round(np.mean(reg_rec[t][1]), 5)
        reg_rec[t][2] = round(np.mean(reg_rec[t][2]), 5)
        reg_rec[t][3] = round(np.mean(reg_rec[t][3]), 5)
        reg_rec[t][4] = round(np.mean(reg_rec[t][4]), 5)
        reg_rec[t][5] = round(np.mean(reg_rec[t][5]), 5)
        reg_rec[t][6] = round(np.mean(reg_rec[t][6]), 5)
        reg_rec[t][7] = round(np.mean(reg_rec[t][7]), 5)
        reg_rec[t][8] = round(np.mean(reg_rec[t][8]), 5)
        reg_rec[t][9] = round(np.mean(reg_rec[t][9]), 5)
        reg_rec[t][10] = round(np.mean(reg_rec[t][10]), 5)
        reg_rec[t][11] = round(np.mean(reg_rec[t][11]), 5)
        reg_rec[t][12] = round(np.mean(reg_rec[t][12]), 5)
        reg_rec[t][13] = round(np.mean(reg_rec[t][13]), 5)
        
        reg_rec[t].append(seeds_dict[t])
        reg_rec[t].append(round(won / denom, 5))

    # set target variables
    for index, row in tour.iterrows():
        wteam = row['WTeamID']
        lteam = row['LTeamID']
        wscore = row['WScore']
        lscore = row['LScore']
        
        # set lower team id as team 1
        if wteam < lteam:
            id_str = str(s) + '_' + str(wteam) + '_' + str(lteam)
            delta = wscore - lscore
            team1 = wteam
            team2 = lteam
        else:
            id_str = str(s) + '_' + str(lteam) + '_' + str(wteam)
            delta = lscore - wscore
            team1 = lteam
            team2 = wteam
        
        train = train.append(
                {'ID': id_str, 'Delta': delta, 'SeedDiff': int(reg_rec[team1][14] - reg_rec[team2][14]),
                 '1FGM': reg_rec[team1][0], '1FGA': reg_rec[team1][1], '1FGM3': reg_rec[team1][2], '1FGA3': reg_rec[team1][3], '1FTM': reg_rec[team1][4], '1FTA': reg_rec[team1][5],
                 '1OR': reg_rec[team1][6], '1DR': reg_rec[team1][7], '1AST': reg_rec[team1][8], '1TO': reg_rec[team1][9], '1STL': reg_rec[team1][10], '1BLK': reg_rec[team1][11], '1PF': reg_rec[team1][12],
                 '1Score': reg_rec[team1][13], '1WRatio': reg_rec[team1][15],
                 '2FGM': reg_rec[team2][0], '2FGA': reg_rec[team2][1], '2FGM3': reg_rec[team2][2], '2FGA3': reg_rec[team2][3], '2FTM': reg_rec[team2][4], '2FTA': reg_rec[team2][5],
                 '2OR': reg_rec[team2][6], '2DR': reg_rec[team2][7], '2AST': reg_rec[team2][8], '2TO': reg_rec[team2][9], '2STL': reg_rec[team2][10], '2BLK': reg_rec[team2][11], '2PF': reg_rec[team2][12],
                 '2Score': reg_rec[team2][13], '2WRatio': reg_rec[team2][15]}, ignore_index = True)

test = pd.DataFrame(columns = ['ID', 'Delta', 'SeedDiff',
                               '1FGM', '1FGA', '1FGM3', '1FGA3', '1FTM', '1FTA',
                               '1OR', '1DR', '1AST', '1TO', '1STL', '1BLK', '1PF',
                               '1Score', '1WRatio',
                               '2FGM', '2FGA', '2FGM3', '2FGA3', '2FTM', '2FTA',
                               '2OR', '2DR', '2AST', '2TO', '2STL', '2BLK', '2PF',
                               '2Score', '2WRatio'])

# create test sets on 2015-2019
for s in stage1_test:
    print(s)
    # regular and tournament games detailed
    reg = raw_reg[raw_reg['Season'] == s]
    tour = raw_tour[raw_tour['Season'] == s]
    conf_tour = raw_conf_tour[raw_conf_tour['Season'] == s]
    
    # conference tourney list
    conf_games = []
    
    for index, row in conf_tour.iterrows():
        game = str(row['DayNum']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])
        conf_games.append(game)
    
    # seeds dictionary for upset variable
    # also, list of unique team IDs in given season's bracket
    seeds = raw_seeds[raw_seeds['Season'] == s]
    seeds_dict = {}
    
    for index, row in seeds.iterrows():
        seeds_dict[row['TeamID']] = int(row['Seed'][1:3])
    
    # reg season stats
    reg_rec = {}

    for t in seeds_dict.keys():
        #['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'AST', 'TO', 'STL', 'BLK', 'PF', 'Delta']
        reg_rec[t] = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]

        won = 0
        denom = 0

        for index, row in reg.iterrows():
            game = str(row['DayNum']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])
            if (row['WTeamID'] == t) and ((row['LTeamID'] in seeds_dict.keys()) | (game in conf_games)):
                reg_rec[t][0].append(row['WFGM'])
                reg_rec[t][1].append(row['WFGA'])
                reg_rec[t][2].append(row['WFGM3'])
                reg_rec[t][3].append(row['WFGA3'])
                reg_rec[t][4].append(row['WFTM'])
                reg_rec[t][5].append(row['WFTA'])
                reg_rec[t][6].append(row['WOR'])
                reg_rec[t][7].append(row['WDR'])
                reg_rec[t][8].append(row['WAst'])
                reg_rec[t][9].append(row['WTO'])
                reg_rec[t][10].append(row['WStl'])
                reg_rec[t][11].append(row['WBlk'])
                reg_rec[t][12].append(row['WPF'])
                reg_rec[t][13].append(row['WScore'] - row['LScore'])
                won += 1
                denom += 1
                
            elif (row['LTeamID'] == t) and ((row['LTeamID'] in seeds_dict.keys()) | (game in conf_games)):
                reg_rec[t][0].append(row['LFGM'])
                reg_rec[t][1].append(row['LFGA'])
                reg_rec[t][2].append(row['LFGM3'])
                reg_rec[t][3].append(row['LFGA3'])
                reg_rec[t][4].append(row['LFTM'])
                reg_rec[t][5].append(row['LFTA'])
                reg_rec[t][6].append(row['LOR'])
                reg_rec[t][7].append(row['LDR'])
                reg_rec[t][8].append(row['LAst'])
                reg_rec[t][9].append(row['LTO'])
                reg_rec[t][10].append(row['LStl'])
                reg_rec[t][11].append(row['LBlk'])
                reg_rec[t][12].append(row['LPF'])
                reg_rec[t][13].append(row['LScore'] - row['WScore'])
                denom += 1

        reg_rec[t][0] = round(np.mean(reg_rec[t][0]), 5)
        reg_rec[t][1] = round(np.mean(reg_rec[t][1]), 5)
        reg_rec[t][2] = round(np.mean(reg_rec[t][2]), 5)
        reg_rec[t][3] = round(np.mean(reg_rec[t][3]), 5)
        reg_rec[t][4] = round(np.mean(reg_rec[t][4]), 5)
        reg_rec[t][5] = round(np.mean(reg_rec[t][5]), 5)
        reg_rec[t][6] = round(np.mean(reg_rec[t][6]), 5)
        reg_rec[t][7] = round(np.mean(reg_rec[t][7]), 5)
        reg_rec[t][8] = round(np.mean(reg_rec[t][8]), 5)
        reg_rec[t][9] = round(np.mean(reg_rec[t][9]), 5)
        reg_rec[t][10] = round(np.mean(reg_rec[t][10]), 5)
        reg_rec[t][11] = round(np.mean(reg_rec[t][11]), 5)
        reg_rec[t][12] = round(np.mean(reg_rec[t][12]), 5)
        reg_rec[t][13] = round(np.mean(reg_rec[t][13]), 5)
        
        reg_rec[t].append(seeds_dict[t])
        reg_rec[t].append(round(won / denom, 5))

    # set target variables
    for index, row in tour.iterrows():
        wteam = row['WTeamID']
        lteam = row['LTeamID']
        wscore = row['WScore']
        lscore = row['LScore']
        
        # set lower team id as team 1
        if wteam < lteam:
            id_str = str(s) + '_' + str(wteam) + '_' + str(lteam)
            delta = wscore - lscore
            team1 = wteam
            team2 = lteam
        else:
            id_str = str(s) + '_' + str(lteam) + '_' + str(wteam)
            delta = lscore - wscore
            team1 = lteam
            team2 = wteam
        
        test = test.append(
                {'ID': id_str, 'Delta': delta, 'SeedDiff': int(reg_rec[team1][14] - reg_rec[team2][14]),
                 '1FGM': reg_rec[team1][0], '1FGA': reg_rec[team1][1], '1FGM3': reg_rec[team1][2], '1FGA3': reg_rec[team1][3], '1FTM': reg_rec[team1][4], '1FTA': reg_rec[team1][5],
                 '1OR': reg_rec[team1][6], '1DR': reg_rec[team1][7], '1AST': reg_rec[team1][8], '1TO': reg_rec[team1][9], '1STL': reg_rec[team1][10], '1BLK': reg_rec[team1][11], '1PF': reg_rec[team1][12],
                 '1Score': reg_rec[team1][13], '1WRatio': reg_rec[team1][15],
                 '2FGM': reg_rec[team2][0], '2FGA': reg_rec[team2][1], '2FGM3': reg_rec[team2][2], '2FGA3': reg_rec[team2][3], '2FTM': reg_rec[team2][4], '2FTA': reg_rec[team2][5],
                 '2OR': reg_rec[team2][6], '2DR': reg_rec[team2][7], '2AST': reg_rec[team2][8], '2TO': reg_rec[team2][9], '2STL': reg_rec[team2][10], '2BLK': reg_rec[team2][11], '2PF': reg_rec[team2][12],
                 '2Score': reg_rec[team2][13], '2WRatio': reg_rec[team2][15]}, ignore_index = True)




# save to csv
train.to_csv('train.csv', sep = ',', index = False)
test.to_csv('test.csv', sep = ',', index = False)

# xgb predict delta
train_feat = train.drop(columns = ['ID', 'Delta'])
train_feat['SeedDiff'] = pd.to_numeric(train_feat['SeedDiff'])

train_label = train[['Delta']]
train_label['Delta'] = pd.to_numeric(train_label['Delta'])

dtrain = xgb.DMatrix(data = train_feat, label = train_label)

test_feat = test.drop(columns = ['ID', 'Delta'])
test_feat['SeedDiff'] = pd.to_numeric(test_feat['SeedDiff'])

dtest = xgb.DMatrix(test_feat)

# raw logloss = 0.56568
param = {'eta': 0.02,
        'min_child_weight': 40,
        'max_depth': 2,
        'subsample': 0.35,
        'colsample_bytree': 0.7,
        'gamma': 5,
        'objective': 'reg:linear',
        'eval_metric': 'auc',
        'seed': 1996,
        'verbosity': 0}

bst = xgb.train(param, dtrain, 1500)

#xgb.plot_importance(bst)

pred = bst.predict(dtest)

test_out = test.copy()[['ID', 'Delta']]

num_add = 2 * np.std(pred) - np.mean(pred)
denom = 4 * np.std(pred)

test_out['pred'] = np.clip((pred + num_add) / denom, .075, .925)

# LogLoss: 0.5529115841947012
for index, row in test_out.iterrows():
    if row['pred'] >= 0.50:
        if row['pred'] < 0.60:
            test_out.at[index, 'pred'] = 0.60
        elif row['pred'] >= 0.85:
            test_out.at[index, 'pred'] = 0.925
    if row['pred'] < 0.50:
        if row['pred'] > 0.40:
            test_out.at[index, 'pred'] = 0.40
        elif row['pred'] <= 0.15:
            test_out.at[index, 'pred'] = 0.075

log_bkt = []
for index, row in test_out.iterrows():
    if row['Delta'] > 0:
        log_bkt.append(math.log(row['pred']))
    else:
        log_bkt.append(math.log(1 - row['pred']))

print('LogLoss:' + str(-np.mean(log_bkt)))

test_out.to_csv('test_out.csv', sep = ',', index = False)




samp = pd.DataFrame(columns = ['ID', 'SeedDiff',
                               '1FGM', '1FGA', '1FGM3', '1FGA3', '1FTM', '1FTA',
                               '1OR', '1DR', '1AST', '1TO', '1STL', '1BLK', '1PF',
                               '1Score', '1WRatio',
                               '2FGM', '2FGA', '2FGM3', '2FGA3', '2FTM', '2FTA',
                               '2OR', '2DR', '2AST', '2TO', '2STL', '2BLK', '2PF',
                               '2Score', '2WRatio'])

# create test sets on 2015-2019
for s in stage1_test:
    print(s)
    # regular and tournament games detailed
    reg = raw_reg[raw_reg['Season'] == s]
    tour = raw_tour[raw_tour['Season'] == s]
    conf_tour = raw_conf_tour[raw_conf_tour['Season'] == s]
    
    # conference tourney list
    conf_games = []
    
    for index, row in conf_tour.iterrows():
        game = str(row['DayNum']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])
        conf_games.append(game)
    
    # seeds dictionary for upset variable
    # also, list of unique team IDs in given season's bracket
    seeds = raw_seeds[raw_seeds['Season'] == s]
    seeds_dict = {}
    
    for index, row in seeds.iterrows():
        seeds_dict[row['TeamID']] = int(row['Seed'][1:3])
    
    # reg season stats
    reg_rec = {}

    for t in seeds_dict.keys():
        #['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'AST', 'TO', 'STL', 'BLK', 'PF', 'Delta']
        reg_rec[t] = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]

        won = 0
        denom = 0

        for index, row in reg.iterrows():
            game = str(row['DayNum']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])
            if (row['WTeamID'] == t) and ((row['LTeamID'] in seeds_dict.keys()) | (game in conf_games)):
                reg_rec[t][0].append(row['WFGM'])
                reg_rec[t][1].append(row['WFGA'])
                reg_rec[t][2].append(row['WFGM3'])
                reg_rec[t][3].append(row['WFGA3'])
                reg_rec[t][4].append(row['WFTM'])
                reg_rec[t][5].append(row['WFTA'])
                reg_rec[t][6].append(row['WOR'])
                reg_rec[t][7].append(row['WDR'])
                reg_rec[t][8].append(row['WAst'])
                reg_rec[t][9].append(row['WTO'])
                reg_rec[t][10].append(row['WStl'])
                reg_rec[t][11].append(row['WBlk'])
                reg_rec[t][12].append(row['WPF'])
                reg_rec[t][13].append(row['WScore'] - row['LScore'])
                won += 1
                denom += 1
                
            elif (row['LTeamID'] == t) and ((row['LTeamID'] in seeds_dict.keys()) | (game in conf_games)):
                reg_rec[t][0].append(row['LFGM'])
                reg_rec[t][1].append(row['LFGA'])
                reg_rec[t][2].append(row['LFGM3'])
                reg_rec[t][3].append(row['LFGA3'])
                reg_rec[t][4].append(row['LFTM'])
                reg_rec[t][5].append(row['LFTA'])
                reg_rec[t][6].append(row['LOR'])
                reg_rec[t][7].append(row['LDR'])
                reg_rec[t][8].append(row['LAst'])
                reg_rec[t][9].append(row['LTO'])
                reg_rec[t][10].append(row['LStl'])
                reg_rec[t][11].append(row['LBlk'])
                reg_rec[t][12].append(row['LPF'])
                reg_rec[t][13].append(row['LScore'] - row['WScore'])
                denom += 1

        reg_rec[t][0] = round(np.mean(reg_rec[t][0]), 5)
        reg_rec[t][1] = round(np.mean(reg_rec[t][1]), 5)
        reg_rec[t][2] = round(np.mean(reg_rec[t][2]), 5)
        reg_rec[t][3] = round(np.mean(reg_rec[t][3]), 5)
        reg_rec[t][4] = round(np.mean(reg_rec[t][4]), 5)
        reg_rec[t][5] = round(np.mean(reg_rec[t][5]), 5)
        reg_rec[t][6] = round(np.mean(reg_rec[t][6]), 5)
        reg_rec[t][7] = round(np.mean(reg_rec[t][7]), 5)
        reg_rec[t][8] = round(np.mean(reg_rec[t][8]), 5)
        reg_rec[t][9] = round(np.mean(reg_rec[t][9]), 5)
        reg_rec[t][10] = round(np.mean(reg_rec[t][10]), 5)
        reg_rec[t][11] = round(np.mean(reg_rec[t][11]), 5)
        reg_rec[t][12] = round(np.mean(reg_rec[t][12]), 5)
        reg_rec[t][13] = round(np.mean(reg_rec[t][13]), 5)
        
        reg_rec[t].append(seeds_dict[t])
        reg_rec[t].append(round(won / denom, 5))

    # set target variables

    # subset season from submission file
    seas = subm[subm.ID.str.contains(str(s))]

    for index, row in seas.iterrows():
        team1 = int(row['ID'][5:9])
        team2 = int(row['ID'][10:14])
        samp = samp.append(
                {'ID': row['ID'], 'SeedDiff': int(reg_rec[team1][14] - reg_rec[team2][14]),
                 '1FGM': reg_rec[team1][0], '1FGA': reg_rec[team1][1], '1FGM3': reg_rec[team1][2], '1FGA3': reg_rec[team1][3], '1FTM': reg_rec[team1][4], '1FTA': reg_rec[team1][5],
                 '1OR': reg_rec[team1][6], '1DR': reg_rec[team1][7], '1AST': reg_rec[team1][8], '1TO': reg_rec[team1][9], '1STL': reg_rec[team1][10], '1BLK': reg_rec[team1][11], '1PF': reg_rec[team1][12],
                 '1Score': reg_rec[team1][13], '1WRatio': reg_rec[team1][15],
                 '2FGM': reg_rec[team2][0], '2FGA': reg_rec[team2][1], '2FGM3': reg_rec[team2][2], '2FGA3': reg_rec[team2][3], '2FTM': reg_rec[team2][4], '2FTA': reg_rec[team2][5],
                 '2OR': reg_rec[team2][6], '2DR': reg_rec[team2][7], '2AST': reg_rec[team2][8], '2TO': reg_rec[team2][9], '2STL': reg_rec[team2][10], '2BLK': reg_rec[team2][11], '2PF': reg_rec[team2][12],
                 '2Score': reg_rec[team2][13], '2WRatio': reg_rec[team2][15]}, ignore_index = True)

samp_feat = samp.drop(columns = ['ID'])
samp_feat['SeedDiff'] = pd.to_numeric(samp_feat['SeedDiff'])

dtest = xgb.DMatrix(samp_feat)

#xgb.plot_importance(bst)

pred = bst.predict(dtest)

samp_out = samp.copy()[['ID']]

num_add = 2 * np.std(pred) - np.mean(pred)
denom = 4 * np.std(pred)

samp_out['Pred'] = np.clip((pred + num_add) / denom, .075, .925)

for index, row in samp_out.iterrows():
    if row['Pred'] >= 0.50:
        if row['Pred'] < 0.60:
            samp_out.at[index, 'Pred'] = 0.60
        elif row['Pred'] >= 0.85:
            samp_out.at[index, 'Pred'] = 0.925
    if row['Pred'] < 0.50:
        if row['Pred'] > 0.40:
            samp_out.at[index, 'Pred'] = 0.40
        elif row['Pred'] <= 0.15:
            samp_out.at[index, 'Pred'] = 0.075

samp_out.to_csv('s1_submission.csv', sep = ',', index = False)

# CV

eta = [.02, .04, .08]
min_child_weight = [20, 30, 40]
max_depth = [2, 3, 4]
subsample = [.35, .50, .65]
colsample_bytree = [.60, .70, .80]
gamma = [0, 5, 10]
num_rounds = [1500, 3000, 5000]

hp_tuning = pd.DataFrame(columns = ['eta', 'min_child_weight', 'max_depth', 'subsample', 'colsample_bytree', 'gamma', 'num_rounds', 'LogLoss'])
count = 0

for a in eta:
    for b in min_child_weight:
        for c in max_depth:
            for d in subsample:
                for e in colsample_bytree:
                    for f in gamma:
                        for g in num_round:

                            param = {'eta': a,
                                    'min_child_weight': b,
                                    'max_depth': c,
                                    'subsample': d,
                                    'colsample_bytree': e,
                                    'gamma': f,
                                    'objective': 'reg:linear',
                                    'eval_metric': 'auc',
                                    'seed': 1996,
                                    'verbosity': 0}

                            bst = xgb.train(param, dtrain, g)

                            #xgb.plot_importance(bst)

                            pred = bst.predict(dtest)

                            test_out = test.copy()[['ID', 'Delta']]

                            num_add = 2 * np.std(pred) - np.mean(pred)
                            denom = 4 * np.std(pred)

                            test_out['pred'] = np.clip((pred + num_add) / denom, .025, .975)

                            log_bkt = []
                            for index, row in test_out.iterrows():
                                if row['Delta'] > 0:
                                    log_bkt.append(math.log(row['pred']))
                                else:
                                    log_bkt.append(math.log(1 - row['pred']))

                            hp_tuning = hp_tuning.append({'eta': a, 'min_child_weight': b, 'max_depth': c, 'subsample': d, 'colsample_bytree': e, 'gamma': f, 'num_rounds': g, 'LogLoss': -np.mean(log_bkt)}, ignore_index = True)
                            count += 1
                            if count % 50 == 0:
                                print(count)

hp_tuning.to_csv('hp_tuning.csv', sep = ',', index = False)