import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Get the data
dataset = pd.read_csv('MNIST.csv')

# Convert X to data type float32 so that X_train and X_test can divide for 255
X = dataset.iloc[:-1000, 1:].values.reshape(-1, 784).astype(np.float32)
y = dataset.iloc[:-1000, 0].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Scale the data
X_train /= 255.
X_test /= 255.
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Import the model
model = SVC()

t0 = datetime.now()
model.fit(X_train, y_train)
print('train duration:', datetime.now() - t0)

t0 = datetime.now()
print('train score:', model.score(X_train, y_train), 'duration:', datetime.now() - t0)

t0 = datetime.now()
print('test score:', model.score(X_test, y_test), 'duration:', datetime.now() - t0)
