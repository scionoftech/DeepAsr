import tensorflow as tf
import os
from deepasr.utils import load_data
import deepasr as asr


def load(directory: str):
    """ Load each component of the CTC pipeline. """

    _network = tf.keras.models.load_model(os.path.join(directory, 'network.h5'))
    _network.load_weights(os.path.join(directory, 'model_weights.h5'))
    _optimizer = load_data(os.path.join(directory, 'optimizer.bin'))
    _alphabet = load_data(os.path.join(directory, 'alphabet.bin'))
    _decoder = load_data(os.path.join(directory, 'decoder.bin'))
    _features_extractor = load_data(
        os.path.join(directory, 'feature_extractor.bin'))
    _multi_gpu_flag = load_data(os.path.join(directory, 'multi_gpu_flag.bin'))
    _sample_rate = load_data(os.path.join(directory, 'sample_rate.bin'))

    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=_alphabet, features_extractor=_features_extractor, model=_network, optimizer=_optimizer,
        decoder=_decoder, sample_rate=_sample_rate, mono=True, multi_gpu=_multi_gpu_flag
    )
    return pipeline
