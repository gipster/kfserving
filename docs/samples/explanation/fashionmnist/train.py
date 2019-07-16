
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Model
from keras.utils import to_categorical
from alibi.explainers import AnchorImage
import joblib


def model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation='softmax')(x)

    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn

np.random.seed(0)
# load data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
print('x_train shape:', X_train.shape, 'y_train shape:', y_train.shape)

# define train and test set
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = np.reshape(X_train, X_train.shape + (1,))
X_test = np.reshape(X_test, X_test.shape + (1,))
print('x_train shape:', X_train.shape, 'x_test shape:', X_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

# define and fit model
cnn = model()
cnn.summary()
cnn.fit(X_train, y_train, batch_size=64, epochs=3)
input_shape = ((1,) + X_train.shape[1:])

# Evaluate the model on test set
score = cnn.evaluate(X_test, y_test, verbose=0)
print('Test accuracy: ', score[1])

# Dump files
cnn.save('model.h5')
joblib.dump(X_train, "train.joblib")
joblib.dump(input_shape, "input_shape.joblib")

