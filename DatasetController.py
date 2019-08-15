import pandas as pd
import numpy as np

"""Import the dataset, shuffle it and take only a fraction for faster processing"""
df = pd.read_csv('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Datasets/Final/complete.csv')
df = df.replace('None', np.nan)
df = df.drop_duplicates()
df = df.dropna()
df = df.reset_index()
df = df.drop(['Usage', 'index'], axis=1)
df.to_csv('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/complete_edited.csv', index=False)