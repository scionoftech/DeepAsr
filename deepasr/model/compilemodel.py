from tensorflow.python.ops import math_ops as tf_math_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
import logging

logger = logging.getLogger('asr.pipeline')

def ctc_batch_cost_custom(y_true, y_pred, input_length, label_length, preprocess_collapse_repeated=False):
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_labels = tf.cast(
        K.ctc_label_dense_to_sparse(y_true, label_length), tf.int32)
    y_pred = tf_math_ops.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())
    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length,
                                       preprocess_collapse_repeated=True),   1)



def ctc_loss(args):
    """ The CTC loss using TensorFlow's `ctc_loss`. """
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost_custom(labels, y_pred, input_length, label_length, True)


def compile_model(_model, _optimizer, label_len=None):
    """ The compiled model means the model configured for training. """

    input_data = _model.inputs[0]
    y_pred = _model.outputs[0]

    # your ground truth data. The data you are going to compare with the model's outputs in training
    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    # the length (in steps, or chars this case) of each sample (sentence) in the y_pred tensor
    input_length = Input(name='input_length', shape=[1], dtype='float32')
    #  the length (in steps, or chars this case) of each sample (sentence) in the y_true
    label_length = Input(name='label_length', shape=[1], dtype='float32')
    output = Lambda(ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    _model = Model(inputs=[input_data, labels, input_length, label_length], outputs=output,
                   name="DeepAsr")
    _model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=_optimizer,
                   metrics=['accuracy'])

    # _model.summary()
    logger.info("Model is successfully compiled")
    return _model
