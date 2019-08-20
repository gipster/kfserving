import numpy as np
import keras
import alibi
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Model
from keras.utils import to_categorical
import joblib
import dill
from keras.wrappers.scikit_learn import KerasClassifier


# load data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)

# define train and test set
np.random.seed(0)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)


def superpixel(image, size=(4, 7)):
    segments = np.zeros([image.shape[0], image.shape[1]])
    row_idx, col_idx = np.where(segments == 0)
    for i, j in zip(row_idx, col_idx):
        segments[i, j] = int((image.shape[1]/size[1]) * (i//size[0]) + j//size[1])
    return segments


# define and  train an cnn model
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

cnn = KerasClassifier(build_fn=model, verbose=1)
# cnn = model()
# cnn.summary()

print('Training cnn ...')
cnn.fit(x_train, y_train, batch_size=64, epochs=1)
print('Training done!')
# score = cnn.evaluate(x_test, y_test, verbose=0)
# print('Test accuracy: ', score[1])

print("Creating an explainer")
predict_fn = lambda x: cnn.predict(x)
image_shape = x_train[0].shape
explainer = alibi.explainers.AnchorImage(predict_fn, image_shape, segmentation_fn=superpixel)

# explainer.fit(x_train)
explainer.predict_fn = None  # Clear explainer predict_fn as its a lambda and will be reset when loaded
with open("explainer.dill", 'wb') as f:
    dill.dump(explainer, f)

print("Saving individual files")
# Dump files - for testing creating an AnchorExplainer from components
# cnn.save('model.h5')
joblib.dump(cnn, "model.joblib")
joblib.dump(x_train, "train.joblib")
joblib.dump(image_shape, "input_shape.joblib")



