import pandas as pd
import os

files = os.listdir('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Datasets/All/')

df_list = []
for i in range(len(files)):
    temp_df = pd.read_csv('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Datasets/All/{}'.format(files[i]))
    df_list.append(temp_df)

df = pd.concat(df_list, axis=0, sort=False, ignore_index=False, join='outer')
df = df.reset_index()
df = df.drop('index', axis=1)

df.to_csv('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/complete.csv', index=False)

