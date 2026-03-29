
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

