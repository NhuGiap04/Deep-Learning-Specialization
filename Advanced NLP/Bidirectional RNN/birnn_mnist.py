# Import the libraries
from keras.layers import Input, LSTM, Bidirectional, Dense, GlobalMaxPooling1D, Concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop

import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Some configuration
HEIGHT = 28
WIDTH = 28
HIDDEN_SIZE = 15
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10


# Load in the data
print("Loading the MNIST data")
mnist = pd.read_csv("../Datasets/MNIST.csv")
X = mnist.iloc[:, 1:].values
Y = mnist.iloc[:, 0].values
print("X_shape is:", X.shape, "and Y_shape is:", Y.shape)

# Normalize the data
X = X/255.

# Reshape the data
X = np.reshape(X, (X.shape[0], HEIGHT, WIDTH))

# Build the model
i = Input(shape=(HEIGHT, WIDTH))
h_init = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(i)
input_init = GlobalMaxPooling1D()(h_init)

# Rotate the image
permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))
i_rotate = permutor(i)
h_rotate = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(i_rotate)
input_rotate = GlobalMaxPooling1D()(h_rotate)

input_ = Concatenate(axis=1)([input_init, input_rotate])
x = Dense(10, activation='softmax')(input_)

model = Model(i, x)

# Build and Train the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'],
)

print('Training model...')
r = model.fit(
    X,
    Y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
