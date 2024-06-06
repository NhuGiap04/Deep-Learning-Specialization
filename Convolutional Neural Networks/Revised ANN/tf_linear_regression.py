import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load in the data
data = pd.read_csv('../Datasets/moore.csv', header=None).values
X = data[:, 0].reshape(-1, 1)  # make it a 2-D array of size N x D where D = 1
Y = data[:, 1]

# Plot the data - it is exponential!
plt.scatter(X, Y)

# Since we want a linear model, let's take the log
Y = np.log(Y)
plt.scatter(X, Y)

# Let's also center the X data so the values are not too large
# We could scale it too but then we'd have to reverse the transformation later
X = X - X.mean()

# Now create our Tensorflow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')


# model.compile(optimizer='adam', loss='mse')

# learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

# Plot the loss
plt.plot(r.history['loss'], label='loss')

# Get the slope of the line
# The slope of the line is related to the doubling rate of transistor count
print(model.layers)  # Note: there is only 1 layer, the "Input" layer doesn't count
print(model.layers[0].get_weights())

# The slope of the line is:
a = model.layers[0].get_weights()[0][0, 0]

# Make sure the line fits our data
Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
