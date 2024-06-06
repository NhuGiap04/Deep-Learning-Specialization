import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

# In Practice, we must deal with columns that have many missing values
df = pd.read_csv('diabetes.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
features = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]
X_train = df_train[features]
y_train = df_train['Outcome']
X_test = df_test[features]
y_test = df_test['Outcome']
model = GaussianNB()
model.fit(X_train, y_train)
print("train accuracy:", model.score(X_train, y_train))
print("test accuracy:", model.score(X_test, y_test))
X = df[features]
y = df['Outcome']
result = cross_val_score(model, X, y)
print(result.mean(), result.std())