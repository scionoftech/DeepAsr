from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import logging

logger = logging.getLogger('asr.pipeline')


def ctc_loss(args):
    """ The CTC loss using TensorFlow's `ctc_loss`. """
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


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
