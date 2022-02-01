from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, InputLayer
from typing import Tuple


def returnConvModel(input_shape: Tuple[int, int], output_shape: int) -> Sequential:
    model = Sequential()
    model.add(InputLayer((input_shape[0], input_shape[1], 1)))
    model.add(Conv2D(32, kernel_size=8, strides=(4, 4), padding="same", input_dim=input_shape))
    model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding="same"))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(output_shape, activation="linear"))
    return model
