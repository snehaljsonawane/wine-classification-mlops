import pandas as pd
from pathlib import Path
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)

df = wine.frame.copy()

df["wine_class"] = df["target"].map(dict(enumerate(wine.target_names)))

Path("data/raw").mkdir(parents=True, exist_ok=True)

df.to_csv("data/raw/wine_v1.csv", index=False)

print("Wine dataset generated successfully")
print("Dataset shape:", df.shape)
print("Saved to: data/raw/wine_v1.csv")
print(df.head())
