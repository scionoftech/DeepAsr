import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.activations import relu


def clipped_relu(x):
    return relu(x, max_value=20)


def get_deepspeech2_v1(input_dim=None, output_dim=29,
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
    conv1 = Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_1')(bn1)
    cbn1 = BatchNormalization(axis=-1, name='CBN_1')(conv1)
    conv2 = Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_2')(cbn1)
    cbn2 = BatchNormalization(axis=-1, name='CBN_2')(conv2)
    conv3 = Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_3')(cbn2)

    # Batch normalize
    x = BatchNormalization(axis=-1, name='BN_2')(conv3)

    # BiRNNs
    # birnn1 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum')(bn2)
    # birnn2 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum')(birnn1)
    # birnn3 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum')(birnn2)
    # birnn4 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_4'), merge_mode='sum')(birnn3)
    # birnn5 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_5'), merge_mode='sum')(birnn4)
    # birnn6 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_6'), merge_mode='sum')(birnn5)
    # birnn7 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_7'), merge_mode='sum')(birnn6)

    # BiRNNs
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

    # Batch normalize
    bn3 = BatchNormalization(axis=-1, name='BN_3')(x)

    dense = TimeDistributed(Dense(1024, activation=clipped_relu, name='FC1'))(bn3)
    y_pred = TimeDistributed(Dense(output_dim, activation='softmax', name='y_pred'))(dense)

    model = Model(inputs=input_data, outputs=y_pred)

    # # your ground truth data. The data you are going to compare with the model's outputs in training
    # labels = Input(name='the_labels', shape=[label_dim], dtype='float32')
    # # the length (in steps, or chars this case) of each sample (sentence) in the y_pred tensor
    # input_length = Input(name='input_length', shape=[1], dtype='float32')
    # #  the length (in steps, or chars this case) of each sample (sentence) in the y_true
    # label_length = Input(name='label_length', shape=[1], dtype='float32')
    # output = Lambda(ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    # model = Model(inputs=[input_data, labels, input_length, label_length], outputs=output, name="deepspeech2pro_v1")
    return model
