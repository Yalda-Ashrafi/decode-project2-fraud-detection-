
import pandas as pd
import numpy as np

df = pd.read_csv("E:/Decode_Intern/fraud-detection/creditcard.csv")

print("Shape:", df.shape)                 # (284807, 31)
print(df.head())
print(df.info())

# Missing values — this dataset has none
print("Missing values:", df.isnull().sum().sum()) 

# The class balance

counts = df["Class"].value_counts()
print(counts)
print("Fraud rate: {:.4f}%".format(100 * counts[1] / len(df)))