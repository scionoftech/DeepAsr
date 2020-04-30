import tensorflow as tf
import os
from deepasr.utils import load_data
from deepasr.pipeline import CTCPipeline
from deepasr.model import compile_model


def load(directory: str):
    """ Load each component of the CTC pipeline. """

    _label_len = load_data(os.path.join(directory, 'label_len.bin'))
    _optimizer = load_data(os.path.join(directory, 'optimizer.bin'))
    _network = tf.keras.models.load_model(os.path.join(directory, 'network.h5'))
    _model = _network
    _model = compile_model(_model, _optimizer, _label_len)
    _model.load_weights(os.path.join(directory, 'model_weights.h5'))
    _alphabet = load_data(os.path.join(directory, 'alphabet.bin'))
    _decoder = load_data(os.path.join(directory, 'decoder.bin'))
    _features_extractor = load_data(
        os.path.join(directory, 'feature_extractor.bin'))
    _multi_gpu_flag = load_data(os.path.join(directory, 'multi_gpu_flag.bin'))
    _sample_rate = load_data(os.path.join(directory, 'sample_rate.bin'))
    _mono = load_data(os.path.join(directory, 'mono.bin'))

    pipeline = CTCPipeline(
        alphabet=_alphabet, features_extractor=_features_extractor, model=_model, optimizer=_optimizer,
        decoder=_decoder, sample_rate=_sample_rate, mono=_mono, label_len=_label_len, multi_gpu=_multi_gpu_flag,
        temp_model=_network
    )
    return pipeline
