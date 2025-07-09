import pandas as pd
import numpy as np



# load train dataset..

train_df= pd.read_csv('train.csv')
print(train_df.shape)
print(train_df.head())