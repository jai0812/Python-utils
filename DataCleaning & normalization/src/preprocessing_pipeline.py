"""
Data Preprocessing Pipeline
- Missing value handling
- Outlier removal (IQR, factor = 2.0)
- Normalization
- Dummy variable encoding
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/hw1.csv")

# ---------------- Problem 1 ----------------
df["age"] = df["age"].fillna(df["age"].mean())
df["duration"] = df["duration"].fillna(df["duration"].median())
df["marital"] = df["marital"].fillna(df["marital"].mode()[0])
df.to_csv("../data/hw1_p1.csv", index=False)

# ---------------- Problem 2 ----------------
def iqr_bounds(s, factor=2.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor*iqr, q3 + factor*iqr

age_l, age_u = iqr_bounds(df["age"])
dur_l, dur_u = iqr_bounds(df["duration"])

df = df[df["age"].between(age_l, age_u) &
        df["duration"].between(dur_l, dur_u)]
df.to_csv("../data/hw1_p2.csv", index=False)

# ---------------- Problem 3 ----------------
df["age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
df["duration"] = (df["duration"] - df["duration"].mean()) / df["duration"].std()
df.to_csv("../data/hw1_p3.csv", index=False)

# ---------------- Problem 4 ----------------
dummies = pd.get_dummies(df["education"], drop_first=True, prefix="education")
df = pd.concat([df.drop(columns=["education"]), dummies], axis=1)
df.to_csv("../data/hw1_p4.csv", index=False)

print("Pipeline completed successfully.")
