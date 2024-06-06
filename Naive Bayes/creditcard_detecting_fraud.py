import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

# In Practice, we must deal with columns that have many missing values
df = pd.read_csv('creditcard.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
