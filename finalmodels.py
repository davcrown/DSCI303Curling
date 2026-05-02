############ DSCI 303 FINAL PROJECT - CURLING ANALYTICS ############

############ PREPROCESSING #############
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.manifold import TSNE

#competition = pd.read_csv("Competition.csv")
#competitors = pd.read_csv("Competitors.csv")
ends = pd.read_csv("Ends.csv")
#games = pd.read_csv("Games.csv")
stones = pd.read_csv("Stones.csv")
#teams = pd.read_csv("Teams.csv")
# datasets

#competition.info() #full
#competitors.info() #full
#ends.info() #powerplay
#games.info() #full
#stones.info() #some stone coords missing, potentially might just remove
#teams.info() #full
#should merge ends info

#stones.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
stones.describe() # Changed indexing to use .iloc
# -1 must be missing val for task and handle and points?
# 4095 indicates stone is taken out of play
results = pd.merge(stones, ends, on=['CompetitionID', 'SessionID', 'GameID', 'EndID', 'TeamID'], suffixes=('_stones', '_ends'), how="left")
results.head(15)

results["PowerPlay"] = results["PowerPlay"].fillna(0)
#results.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

shot_info = results.iloc[:, 7:]
shot_info.describe()

shot_info['Task'] = shot_info['Task'].transform(str)
shot_info['Handle'] = shot_info['Handle'].transform(str)


shot_info['TimeOut'] = shot_info['TimeOut'].fillna(0)
shot_info['PowerPlay'] = shot_info['PowerPlay'].replace([0,1,2],['None','Left','Right'])
#shot_info.info()


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

    current_stone_distances = []

    for idx in range(1, len(shot_info)):
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

    distances[distance_col_name] = current_stone_distances

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

#print(distances.tail())
print(distances.shape)

distances['Task'] = shot_info['Task']
task_mapping = {
    "0": "Draw",
    "1": "Front",
    "2": "Guard",
    "3": "Raise / Tap-back",
    "4": "Wick / Soft Peeling",
    "5": "Freeze",
    "6": "Take-out",
    "7": "Hit and Roll",
    "8": "Clearing",
    "9": "Double Take-out",
    "10": "Promotion Take-out",
    "11": "through",
    "13": "no statistics"
}

distances['Task'] = distances['Task'].map(task_mapping)
print(distances["Task"].head())
distances['Handle'] = shot_info['Handle']
distances['Points'] = shot_info['Points']
distances['TimeOut'] = shot_info['TimeOut']
distances['PowerPlay'] = shot_info['PowerPlay']
distances['Result'] = shot_info['Result']

# Identify all columns that represent stone distances
stone_distance_cols = [col for col in distances.columns if 'stone_' in col and '_distance' in col]


distances['OverallAvgStoneDistance'] = distances[stone_distance_cols].mean(axis=1)

avgbytask = distances.groupby('Task')['OverallAvgStoneDistance'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x=avgbytask['Task'], y=avgbytask['OverallAvgStoneDistance'], palette='viridis')
plt.xlabel('Task')
plt.ylabel('Average Distance')
plt.title('Average Stones Distance Moved by Each Task')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Remove undefined points values
distances = distances[distances['Points']!=-1]
distances = distances[distances['Points']!=9]
distances = distances[distances['Points']!=10]
distances['StonesInPlay'] = np.sum(np.isnan(distances), 1)

# Inspect the data types and missing values in the 'distances' DataFrame
print(distances.info())
#not sure if median is best choice but it is a common method for handling missing values in numerical data
for col in stone_distance_cols:
    distances[col] = distances[col].fillna(distances[col].median())
distances['OverallAvgStoneDistance'] = distances['OverallAvgStoneDistance'].fillna(distances['OverallAvgStoneDistance'].median())

# Identify categorical columns for one-hot encoding
categorical_cols = ['Task', 'Handle', 'PowerPlay']

# Perform one-hot encoding
distances_encoded = pd.get_dummies(distances, columns=categorical_cols, dummy_na=False)

#print("\nDataFrame after encoding categorical features:")
#print(distances_encoded.head())
print(distances_encoded.shape)

distances_num = distances.select_dtypes(include = 'number')

sns.heatmap(distances_num.corr(), cmap='coolwarm')
plt.show()

########## K MEANS CLUSTERING - clustering model ##########
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#compile/standardize features based on characteristics
cluster_df = distances_encoded.drop(columns=['Result', 'Points'], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

#elbow plot
inertias = []
K_range = range(2, 51) #approx 20 clusters is ideal based on plot

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Plot for K-means Clustering")
plt.show()

#fit k means
k = 20
kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

#add labels to df
distances_clustered = distances.copy()
distances_clustered['Cluster'] = clusters

#pca visualization
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2','PC3','PC4','PC5',
                                      'PC6','PC7','PC8','PC9','PC10'])
pca_df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
for c in sorted(pca_df['Cluster'].unique()):
    subset = pca_df[pca_df['Cluster'] == c]
    plt.scatter(subset['PC1'], subset['PC2'], alpha=0.6, label=f'Cluster {c}')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-means Clusters Visualized with PCA")
plt.legend()
plt.show()

#color code by points
if 'Points' in distances.columns:
    pca_df['Points'] = distances['Points'].reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Points'],
        alpha=0.6
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA View of Shots Colored by Points")
    plt.colorbar(scatter, label='Points')
    plt.show()

#cluster insights - more interpretable than pca visualization
#numeric summary
numeric_cols = distances_clustered.select_dtypes(include=np.number).columns.tolist()
cluster_summary = distances_clustered.groupby('Cluster')[numeric_cols].mean()
print("\nCluster mean summary:")
print(cluster_summary)

#most common task in each cluster
if 'Task' in distances_clustered.columns:
    task_summary = distances_clustered.groupby('Cluster')['Task'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan
    )
    print("\nMost common Task in each cluster:")
    print(task_summary)

#most common handle in each cluster
if 'Handle' in distances_clustered.columns:
    handle_summary = distances_clustered.groupby('Cluster')['Handle'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan
    )
    print("\nMost common Handle in each cluster:")
    print(handle_summary)

#most common power play type in each cluster
if 'PowerPlay' in distances_clustered.columns:
    powerplay_summary = distances_clustered.groupby('Cluster')['PowerPlay'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan
    )
    print("\nMost common PowerPlay in each cluster:")
    print(powerplay_summary)

#number of shots per cluster
print("\nNumber of shots in each cluster:")
print(distances_clustered['Cluster'].value_counts().sort_index())

### train test split and encoding needed for supervised learning ###
# variables for score classification
# keep only valid scoring values
valid_mask = distances_encoded['Points'].isin([0, 1, 2, 3, 4])
X = distances_encoded.loc[valid_mask].drop(columns=['Points', 'Result'], errors='ignore')
y = distances_encoded.loc[valid_mask, 'Points']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_full = scaler.fit_transform(X)



########## LOGISTIC REGRESSION - baseline model ##########
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(class_weight="balanced")
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))



# Oversampling to  balance the Points
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(distances_encoded.drop(columns=['Points', 'Result']), y)

np.unique(y_resampled, return_counts=True)# Oversampling to  balance the Points
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(distances_encoded.drop(columns=['Points', 'Result']), y)

np.unique(y_resampled, return_counts=True)

########## MLP NEURAL NETWORK - neural network model ##########
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

#manually assign weights 
#doing it by data proportions caused model to overly bias class 4
weight_dict = {0: 1.5, 1: 3.0, 2: 1.2, 3: 1.1, 4: 0.8}

#convert labels to integers rather than strings
y_train_nn = y_train.astype(int)
y_test_nn = y_test.astype(int)

#very low learning rate because data is complex and unbalanced
opt = keras.optimizers.Adam(learning_rate=0.0001) 
mlp = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='leaky_relu'), #leaky relu activation to prevent class 3 from disappearing
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
])

mlp.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#implement early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = mlp.fit(
    X_train_scaled, y_train_nn,
    validation_split=0.2,
    epochs=100,
    batch_size=64, #large batch size to stabilize model
    class_weight=weight_dict,
    callbacks=[early_stop],
    verbose=1
)

mlp_pred_probs = mlp.predict(X_test_scaled)
mlp_pred = np.argmax(mlp_pred_probs, axis=1)

print("Accuracy:", accuracy_score(y_test_nn, mlp_pred))
print(classification_report(y_test_nn, mlp_pred))

#training/validation accuracy and loss curves
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(14, 5))

    #accuracy curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    #loss curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)


########## TSNE VISUALIZATION - other model ##########
#pretty sure this is still the old tsne code before updating it, so just replace if needed
df_std = StandardScaler().fit_transform(distances_encoded)

tsne = TSNE(n_components=2, random_state=0, perplexity=75)

curl_tsne = tsne.fit_transform(df_std)
curl_tsne = pd.DataFrame(curl_tsne, columns=['x', 'y'])
y = distances_encoded['Points']
curl_tsne['Points'] = y.reset_index(drop=True)

print(curl_tsne.shape)

plt.figure(figsize=(16,10))
scatter = sns.scatterplot(
    x="x", y="y",
    hue="Points",
    palette="viridis",
    data=curl_tsne,
    alpha=0.5
)
plt.title('t-SNE visualization of Curling Data colored by Points')
plt.colorbar(scatter.get_children()[0], label='Points')
plt.show()


########## MODEL COMPARISONS ##########
results = pd.DataFrame({
    'Model': ['Logistic', 'Random Forest', 'MLP'],
    'Accuracy': [
        accuracy_score(y_test, logit_pred),
        #insert rf accuracy here
        accuracy_score(y_test_nn, mlp_pred)
    ]
})

print("\nMODEL COMPARISON")
print(results.sort_values(by='Accuracy', ascending=False))


## Random Forest Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# Scale the features (important for some models, good practice for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200], # Number of trees
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None], # Maximum depth of the tree
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
    'class_weight': ['balanced'] # Keep class weight as it was useful
}

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier,
                         param_grid=param_grid,
                         cv=3, # Using 3-fold cross-validation for grid search
                         n_jobs=-1, # Use all available cores
                         verbose=2, # Verbosity level
                         scoring='f1_weighted') # Or 'f1_weighted', 'roc_auc' for imbalanced data

# Fit GridSearchCV to the training data
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# Evaluate the best model on the test set
best_rf_model = grid_search.best_estimator_
test_accuracy = best_rf_model.score(X_test_scaled, y_test)
print(f"Test set accuracy with best parameters: {test_accuracy:.4f}")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler

# Import KerasClassifier for wrapping Keras models
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

scaler = StandardScaler()
X_train_scaled_for_clf = scaler.fit_transform(X_train)

logreg_model = LogisticRegression(max_iter=1000, solver='liblinear')

# Get the best Random Forest Classifier model from GridSearchCV
# Assuming best_rf_model is available in the kernel from cell 5bf4a735

# Define the MLP model building function for KerasClassifier
def build_mlp_model():
    opt = Adam(learning_rate=0.001)
    mlp = Sequential([
        Dense(128, activation='leaky_relu', input_shape=(X_train_scaled_for_clf.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(5, activation='softmax') # 5 classes for 'Points' (0,1,2,3,4)
    ])
    mlp.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return mlp

# Create KerasClassifier instance for the MLP model
mlp_model_wrapped = KerasClassifier(model=build_mlp_model, epochs=100, batch_size=64, verbose=0,
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Define the models for comparison
classification_models = {
    'Logistic Regression': logreg_model,
    'Tuned Random Forest Classifier': best_rf_model,
    'MLP Neural Network': mlp_model_wrapped
}

# Perform 5-fold cross-validation for each classification model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
clf_results_scores = {}

print("Cross-validation Accuracy scores for Classification Models:")
for name, model in classification_models.items():
    scores = cross_val_score(model, X_train_scaled_for_clf, y_train, cv=kf, scoring='accuracy')
    clf_results_scores[name] = scores
    print(f"{name}: Mean Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Prepare data for Tukey's HSD test
# Combine all scores into a single array and create group labels
all_scores = np.concatenate(list(clf_results_scores.values()))
model_names = []
for name, scores in clf_results_scores.items():
    model_names.extend([name] * len(scores))
