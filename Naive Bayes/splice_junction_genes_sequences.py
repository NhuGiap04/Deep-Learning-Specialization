import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB

# In Practice, we must deal with columns that have many missing values
df = pd.read_csv('dna.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess the data (convert it to the suitable shape)
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values
