import pandas as pd
df = pd.read_csv("data/processed/metadata_2k.csv")
print(df["label"].value_counts())
