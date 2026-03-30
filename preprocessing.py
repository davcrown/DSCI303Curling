
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

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

distances = pd.DataFrame()
all_distances = [] # Initialize an empty list to store distances

for stone in range(1,13):
    # Create column names for the current stone
    stone_x_col = 'stone_'+str(stone)+'_x'
    stone_y_col = 'stone_'+str(stone)+'_y'
    distance_col_name = 'stone_'+str(stone)+'_distance'
    
    # Temporary list to store distances for the current stone across all shots
    current_stone_distances = []
    
    for idx in range(1, len(shot_info)):
        # Check if coordinates for the current stone are valid (not 4095)
        if shot_info.loc[idx, stone_x_col] != 4095 and shot_info.loc[idx, stone_y_col] != 4095:
            dist = distance(
                shot_info.loc[idx, stone_x_col],
                shot_info.loc[idx, stone_y_col],
                shot_info.loc[idx-1, stone_x_col],
                shot_info.loc[idx-1, stone_y_col]
            )
            current_stone_distances.append(dist)
        else:
            current_stone_distances.append(np.nan) # Append NaN if stone is out of play

    current_stone_distances.insert(0, np.nan) # Or 0, depending on desired behavior for the first shot
    
    # Add the list of distances for the current stone as a column to the distances DataFrame    distances[distance_col_name] = current_stone_distances

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the average distance for each stone
average_distances = distances.mean()

# Plot the average distances
plt.figure(figsize=(12, 6))
sns.barplot(x=average_distances.index, y=average_distances.values, palette='viridis')
plt.xlabel('Stone Number')
plt.ylabel('Average Distance')
plt.title('Average Distance Moved by Each Stone per Shot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(distances.tail())
print(distances.shape)

distances['Task'] = shot_info['Task']
distances['Handle'] = shot_info['Handle']
distances['Points'] = shot_info['Points']
distances['TimeOut'] = shot_info['TimeOut']
distances['PowerPlay'] = shot_info['PowerPlay']
distances['Result'] = shot_info['Result']

# Identify all columns that represent stone distances
stone_distance_cols = [col for col in distances.columns if 'stone_' in col and '_distance' in col]

# Calculate the mean of all stone distances for each row (shot)
# Use .mean(axis=1) to average across columns for each row
distances['OverallAvgStoneDistance'] = distances[stone_distance_cols].mean(axis=1)

# Group by 'Task' and calculate the mean of the newly created 'OverallAvgStoneDistance'
avgbytask = distances.groupby('Task')['OverallAvgStoneDistance'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x=avgbytask['Task'], y=avgbytask['OverallAvgStoneDistance'])
plt.xlabel('Task')
plt.ylabel('Average Distance')
plt.title('Average Distance Moved by Each Task')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Inspect the data types and missing values in the 'distances' DataFrame
print(distances.info())

for col in stone_distance_cols:
    distances[col] = distances[col].fillna(distances[col].median())
distances['OverallAvgStoneDistance'] = distances['OverallAvgStoneDistance'].fillna(distances['OverallAvgStoneDistance'].median())

# Identify categorical columns for one-hot encoding
categorical_cols = ['Task', 'Handle', 'Points', 'PowerPlay']

# Perform one-hot encoding
distances_encoded = pd.get_dummies(distances, columns=categorical_cols, dummy_na=False)

print("\nDataFrame after encoding categorical features:")
print(distances_encoded.head())
print(distances_encoded.shape)
