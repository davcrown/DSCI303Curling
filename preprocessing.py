
import pandas as pd
competition = pd.read_csv("Competition.csv")
competitors = pd.read_csv("Competitors.csv")
ends = pd.read_csv("Ends.csv")
games = pd.read_csv("Games.csv")
stones = pd.read_csv("Stones.csv")
teams = pd.read_csv("Teams.csv")
# datasets

competition.info() #full
competitors.info() #full
ends.info() #powerplay
games.info() #full
stones.info() #some stone coords missing, potentially might just remove
teams.info() #full
#should merge ends info

results = pd.merge(stones, ends, on=['CompetitionID', 'SessionID', 'GameID', 'EndID', 'TeamID'], suffixes=('_stones', '_ends'), how="left")
results.head(15)

results["PowerPlay"] = results["PowerPlay"].fillna(0)

results.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

shot_info = results.iloc[:, 7:]
shot_info.describe()

shot_info['Task'] = shot_info['Task'].to_numpy()
shot_info['Task'] = shot_info['Task'].replace(-1, np.nan)
shot_info['Task'] = str(shot_info['Task'])
shot_info['Handle'] = str(shot_info['Handle'])
shot_info['Points'] = str(shot_info['Points'])
shot_info['Points'] = shot_info['Points'].replace(-1, np.nan)
shot_info['TimeOut'] = shot_info['TimeOut'].fillna(0)
shot_info['PowerPlay'] = shot_info['PowerPlay'].replace([0,1,2],['None','Left','Right'])
shot_info.info()

import seaborn as sns
import matplotlib.pyplot as plt
shot_info_num = shot_info.select_dtypes(include = 'number')

sns.heatmap(shot_info_num.corr(), cmap='coolwarm')
plt.show()
golden_features = shot_info_num.corr()['Result'].sort_values(ascending=False)[1:6].index.tolist()
golden_features

