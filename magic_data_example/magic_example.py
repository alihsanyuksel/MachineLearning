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