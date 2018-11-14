import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D

if 1:
    # prepare data

    import sys
    sys.path.append(r'../datasets')

    from load_moonboard import load_moonboard
    (x_train, y_train), (x_test, y_test) = load_moonboard(r'../datasets/moonboard.npz')

    x_train = x_train.reshape((-1,18,11,1))

    #np.unique(y_test)
    #Out[20]: array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16], dtype=uint8)

    #np.unique(y_train)
    #Out[21]: array([ 0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16], dtype=uint8)

    # to one-hot rep
    n_cls = 17
    Y_train = np.eye(n_cls)[y_train]
    Y_test = np.eye(n_cls)[y_test]


if 1:
    # prepare and train model
    # see 'Machine Learning Methods for Climbing Route Classification'

    batch_size = 128
    inputs = Input(shape=(18,11,1)) #HWC

    x0 = Conv2D(filters=4, kernel_size=(11,7), strides=(1,1), padding='same', input_shape=(18,11,1))(inputs)
    x0 = Flatten()(x0)
    x1 = Flatten()(inputs)
    x = concatenate([x0, x1])
    x = Dense(units=(5*18*11), activation='softmax')(x)
    x = Dense(units=50, activation='softmax')(x)
    outputs = Dense(units=n_cls, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=Y_train, batch_size=batch_size, epochs=10)


'''
Using TensorFlow backend.
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 18, 11, 1)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 18, 11, 4)    312         input_1[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 792)          0           conv2d_1[0][0]
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 198)          0           input_1[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 990)          0           flatten_1[0][0]
                                                                 flatten_2[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 990)          981090      concatenate_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 50)           49550       dense_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 17)           867         dense_2[0][0]
==================================================================================================
Total params: 1,031,819
Trainable params: 1,031,819
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/10
9524/9524 [==============================] - 7s 731us/step - loss: 2.7509 - acc: 0.1378
Epoch 2/10
9524/9524 [==============================] - 6s 652us/step - loss: 2.6173 - acc: 0.1664
Epoch 3/10
9524/9524 [==============================] - 6s 615us/step - loss: 2.5058 - acc: 0.3034
Epoch 4/10
9524/9524 [==============================] - 6s 612us/step - loss: 2.4124 - acc: 0.3034
Epoch 5/10
9524/9524 [==============================] - 6s 608us/step - loss: 2.3348 - acc: 0.3034
Epoch 6/10
9524/9524 [==============================] - 6s 616us/step - loss: 2.2706 - acc: 0.3034
Epoch 7/10
9524/9524 [==============================] - 6s 614us/step - loss: 2.2160 - acc: 0.3034
Epoch 8/10
9524/9524 [==============================] - 6s 609us/step - loss: 2.1707 - acc: 0.3034
Epoch 9/10
9524/9524 [==============================] - 6s 625us/step - loss: 2.1327 - acc: 0.3034
Epoch 10/10
9524/9524 [==============================] - 6s 633us/step - loss: 2.0993 - acc: 0.3034
'''

