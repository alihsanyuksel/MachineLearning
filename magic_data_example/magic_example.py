import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)

# Display first few rows
print(df.head())
# print(df.ifo())
# print(df.describe())
# df.to_csv("output.csv", index=False)

# print(df["class"].unique())
df["class"] = (df["class"] == "g").astype(int)

print(df.head())

#histograms for features
""" for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show() """

#train, validation, test data sets

train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y
print("train")
print(train_df)

print(len(train_df[train_df["class"]==1])) #gamma
print(len(train_df[train_df["class"]==0])) #hadron

train, X_train, y_train = scale_dataset(train_df, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid_df, oversample=False)
test, X_test, y_test = scale_dataset(test_df, oversample=False)

print(len(y_train))

print(sum(y_train == 1))
print(sum(y_train == 0))

#K-Nearest-Neighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(y_pred)

#precision(in model prediction correct ones), recall (in all desired label correct ones), f1-score: balanced precision and recall
print(classification_report(y_test, y_pred))

#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print(classification_report(y_test, y_pred))