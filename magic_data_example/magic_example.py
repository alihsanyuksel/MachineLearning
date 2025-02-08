import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()