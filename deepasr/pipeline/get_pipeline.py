import tensorflow as tf
import os
from deepasr.utils import load_data
import deepasr as asr


def load(directory: str):
    """ Load each component of the CTC pipeline. """
    _model = asr.model.deepspeech2_v1.get_deepspeech2(
        input_dim=161,
        output_dim=29,
        rnn_units=800,
        is_mixed_precision=True
    )
    optimizer = tf.keras.optimizers.Adam(
        lr=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    _model.load_weights(os.path.join(directory, 'model_weights.h5'))
    _alphabet = load_data(os.path.join(directory, 'alphabet.bin'))
    _decoder = load_data(os.path.join(directory, 'decoder.bin'))
    _features_extractor = load_data(
        os.path.join(directory, 'feature_extractor.bin'))

    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=_alphabet, features_extractor=_features_extractor, model=_model, optimizer=optimizer, decoder=_decoder,
        multi_gpu=False
    )
    return pipeline
import tensorflow as tf
import os
from deepasr.utils import load_data
import deepasr as asr


def load(directory: str):
    """ Load each component of the CTC pipeline. """
    _model = asr.model.deepspeech2_v1.get_deepspeech2(
        input_dim=161,
        output_dim=29,
        rnn_units=800,
        is_mixed_precision=True
    )
    optimizer = tf.keras.optimizers.Adam(
        lr=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    _model.load_weights(os.path.join(directory, 'model_weights.h5'))
    _alphabet = load_data(os.path.join(directory, 'alphabet.bin'))
    _decoder = load_data(os.path.join(directory, 'decoder.bin'))
    _features_extractor = load_data(
        os.path.join(directory, 'feature_extractor.bin'))

    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=_alphabet, features_extractor=_features_extractor, model=_model, optimizer=optimizer, decoder=_decoder,
        multi_gpu=False
    )
    return pipeline
