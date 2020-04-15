import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def get_deepspeech2_v2(input_dim=None, output_dim=29,
                       is_mixed_precision=True, random_state=1) -> keras.Model:
    """

    input_dim: int i wielokrotność 4
    output_dim: licba liter w słowniku

    """
    if is_mixed_precision:
        policy = mixed_precision.Policy('float32')  # mixed_float16 | float32
        mixed_precision.set_policy(policy)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # the input
    input_data = Input(name='the_input', shape=(None, input_dim), dtype='float32')

    # Add 4th dimension [batch, time, frequency, channel]
    x = Lambda(keras.backend.expand_dims,
               arguments=dict(axis=-1))(input_data)
    x = Conv2D(filters=32,
               kernel_size=[11, 41],
               strides=[2, 2],
               padding='same',
               use_bias=False,
               name='conv_1')(x)
    x = BatchNormalization(name='conv_1_bn')(x)
    x = ReLU(name='conv_1_relu')(x)

    x = Conv2D(filters=32,
               kernel_size=[11, 21],
               strides=[1, 2],
               padding='same',
               use_bias=False,
               name='conv_2')(x)
    x = BatchNormalization(name='conv_2_bn')(x)
    x = ReLU(name='conv_2_relu')(x)

    x = Conv2D(filters=32,
               kernel_size=[11, 21],
               strides=[1, 2],
               padding='same',
               use_bias=False,
               name='conv_3')(x)
    x = BatchNormalization(name='conv_3_bn')(x)
    x = ReLU(name='conv_3_relu')(x)
    # We need to squeeze to 3D tensor. Thanks to the stride in frequency
    # domain, we reduce the number of features four times for each channel.
    x = Reshape([-1, input_dim // 4 * 32])(x)
    # x = Reshape([-1, input_dim])(x)

    for i in [1, 2, 3, 4, 5, 6, 7]:
        recurrent = GRU(units=800,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        use_bias=True,
                        return_sequences=True,
                        reset_after=True,
                        name=f'gru_{i}')
        x = Bidirectional(recurrent,
                          name=f'bidirectional_{i}',
                          merge_mode='concat')(x)
        x = Dropout(rate=0.5)(x) if i < 7 else x  # Only between

    # Return at each time step logits along characters. Then CTC
    # computation is more stable, in contrast to the softmax.
    x = TimeDistributed(Dense(units=800 * 2), name='dense_1')(x)
    x = ReLU(name='dense_1_relu')(x)
    x = Dropout(rate=0.5)(x)
    y_pred = TimeDistributed(Dense(units=output_dim, activation='softmax', name='y_pred'))(x)

    model = Model(inputs=input_data, outputs=y_pred)
    return model
