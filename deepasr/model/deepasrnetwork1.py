import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def get_deepasrnetwork1(input_dim=None, output_dim=29,
                        is_mixed_precision=True, random_state=1) -> keras.Model:
    """

    input_dim: int i wielokrotność 4
    output_dim: licba liter w słowniku

    """
    if is_mixed_precision:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # the input
    input_data = Input(name='the_input', shape=(None, input_dim), dtype='float32')

    # Batch normalize
    bn1 = BatchNormalization(axis=-1, name='BN_1')(input_data)

    # 1D Convs
    conv = Conv1D(filters=220, kernel_size=11, strides=2, padding='valid', activation='relu', name='Conv1D_1')(bn1)
    conv = BatchNormalization(name="CNBN_1")(conv)
    conv1 = Conv1D(filters=220, kernel_size=11, strides=2, padding='valid', activation='relu', name='Conv1D_2')(conv)
    conv1 = BatchNormalization(name="CNBN_2")(conv1)

    # RNN
    gru_1 = GRU(512, return_sequences=True, name='gru_1')(conv1)
    gru_2 = GRU(512, return_sequences=True, go_backwards=True, name='gru_2')(conv1)

    # merge tow gpu ouputs
    merged = concatenate([gru_1, gru_2])
    # Batch normalize
    bn2 = BatchNormalization(name="BN_2")(merged)

    dense = TimeDistributed(Dense(30))(bn2)
    y_pred = TimeDistributed(Dense(output_dim, activation='softmax', name='y_pred'))(dense)

    model = Model(inputs=input_data, outputs=y_pred)

    return model
