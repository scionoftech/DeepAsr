import os
import logging
from typing import List
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, wait
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import sys

sys.path.append("..")
from deepasr.pipeline import Pipeline
from deepasr.augmentation import Augmentation
from deepasr.decoder import Decoder
from deepasr.features import FeaturesExtractor
from deepasr.vocab import Alphabet
from deepasr.utils import read_audio, save_data

logger = logging.getLogger('asr.pipeline')


class CTCPipeline(Pipeline):
    """
    The pipeline is responsible for connecting a neural network model with
    all non-differential transformations (features extraction or decoding),
    and dependencies. Components are independent.
    """

    def __init__(self,
                 alphabet: Alphabet,
                 features_extractor: FeaturesExtractor,
                 model: keras.Model,
                 optimizer: keras.optimizers.Optimizer,
                 decoder: Decoder,
                 sample_rate: 16000,
                 mono: True,
                 multi_gpu: bool = True):
        self._alphabet = alphabet
        self._model_cpu = model
        self._optimizer = optimizer
        self._decoder = decoder
        self._features_extractor = features_extractor
        self.sample_rate = sample_rate
        self.mono = mono
        self._model = self.distribute_model(model) if multi_gpu else model

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    @property
    def features_extractor(self) -> FeaturesExtractor:
        return self._features_extractor

    @property
    def model(self) -> keras.Model:
        return self._model_cpu

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    def preprocess(self,
                   data: List[np.ndarray],
                   is_extracted: bool,
                   augmentation: Augmentation) -> np.ndarray:
        """ Preprocess batch data to format understandable to a model. """

        if is_extracted:  # then just align features
            features = FeaturesExtractor.align(data)
        else:
            features = self._features_extractor(data)
        features = augmentation(features) if augmentation else features
        # labels = self._alphabet.get_batch_labels(transcripts)
        return features

    def ctc_loss(self, args):
        """ The CTC loss using TensorFlow's `ctc_loss`. """
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def compile_model(self, label_dim):
        """ The compiled model means the model configured for training. """
        # your ground truth data. The data you are going to compare with the model's outputs in training
        input_data = self._model.inputs[0]
        y_pred = self._model.outputs[0]

        labels = Input(name='the_labels', shape=[label_dim], dtype='float32')
        # the length (in steps, or chars this case) of each sample (sentence) in the y_pred tensor
        input_length = Input(name='input_length', shape=[1], dtype='float32')
        #  the length (in steps, or chars this case) of each sample (sentence) in the y_true
        label_length = Input(name='label_length', shape=[1], dtype='float32')
        output = Lambda(self.ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        self._model = Model(inputs=[input_data, labels, input_length, label_length], outputs=output,
                            name="deepspeech2")
        self._model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=self._optimizer,
                            metrics=['accuracy'])

        self._model.summary()
        logger.info("Model is successfully compiled")

    def fit_iter(self,
                 train_dataset: pd.DataFrame,
                 augmentation: Augmentation = None,
                 prepared_features: bool = False,
                 iter_num: int = 1000,
                 batch_size: int = 32,
                 epochs: int = 3,
                 **kwargs) -> keras.callbacks.History:
        """ Get ready data, compile and train a model. """

        history = keras.callbacks.History()

        audios = train_dataset['path'].to_list()

        labels = self._alphabet.get_batch_labels(train_dataset['transcripts'].to_list())

        transcripts = train_dataset['transcripts'].to_list()

        train_len_ = len(transcripts)

        if not self._model.optimizer:  # a loss function and an optimizer
            self.compile_model(labels.shape[1])  # have to be set before the training

        for i in range(iter_num):
            train_index = random.sample(range(train_len_ - 25), batch_size)

            x_train = [audios[i] for i in train_index]

            y_train = [labels[i] for i in train_index]

            y_trans = [transcripts[i] for i in train_index]

            train_inputs = self.wrap_preprocess(x_train,
                                                y_train,
                                                y_trans, augmentation, prepared_features)

            outputs = {'ctc': np.zeros([batch_size])}

            # print(train_inputs['the_input'].shape)
            # print(train_inputs['the_labels'].shape)
            # print(train_inputs['input_length'].shape)
            # print(train_inputs['label_length'].shape)
            # print(train_inputs['input_length'])
            # print(train_inputs['label_length'])

            if i % 100 == 0:
                print(i)
                history = self._model.fit(train_inputs, outputs,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1, **kwargs)
            else:
                history = self._model.fit(train_inputs, outputs,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=0, **kwargs)

        return history

    def fit_generator(self, train_dataset: pd.DataFrame,
                      shuffle: bool = True,
                      augmentation: Augmentation = None,
                      prepared_features: bool = False,
                      batch_size: int = 32,
                      epochs: int = 3,
                      verbose: int = 1,
                      **kwargs) -> keras.callbacks.History:

        """ Get ready data, compile and train a model. """

        audios = train_dataset['path'].to_list()

        labels = self._alphabet.get_batch_labels(train_dataset['transcripts'].to_list())

        transcripts = train_dataset['transcripts'].to_list()

        train_len_ = len(transcripts)

        if not self._model.optimizer:  # a loss function and an optimizer
            self.compile_model(labels.shape[1])  # have to be set before the training

        train_gen = self.get_generator(audios, labels, transcripts,
                                       batch_size, shuffle, augmentation, prepared_features)

        return self._model.fit(train_gen, epochs=epochs,
                               steps_per_epoch=train_len_ // batch_size, verbose=verbose, **kwargs)

    def get_generator(self, audio_paths: List[str], texts: np.array, transcripts: List[str], batch_size: int = 32,
                      shuffle: bool = True, augmentation: Augmentation = None,
                      prepared_features: bool = False):
        """ Data Generator """

        def generator():
            num_samples = len(audio_paths)
            while True:
                x = list()
                y = list()
                if shuffle:
                    temp = list(zip(audio_paths, texts))
                    random.Random(123).shuffle(temp)
                    x, y = list(zip(*temp))

                pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
                future = pool.submit(self.wrap_preprocess,
                                     x[:batch_size],
                                     y[:batch_size], transcripts[:batch_size], augmentation, prepared_features)
                for offset in range(batch_size, num_samples, batch_size):
                    wait([future])
                    batch = future.result()
                    future = pool.submit(self.wrap_preprocess,
                                         x[offset: offset + batch_size],
                                         y[offset: offset + batch_size], transcripts[offset:offset + batch_size],
                                         augmentation, prepared_features)
                    yield batch, {'ctc': np.zeros([batch_size])}

        return generator()

    def predict(self, audio: str, **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. """
        in_features = self._features_extractor([read_audio(audio, sample_rate=self.sample_rate, mono=self.mono)])
        batch_logits = self._model.predict(in_features, **kwargs)
        decoded_labels = self._decoder(batch_logits)
        predictions = self._alphabet.get_batch_transcripts(decoded_labels)
        return predictions

    def wrap_preprocess(self, audios: List[str], the_labels: List[np.array], transcripts: List[str],
                        augmentation: Augmentation = None,
                        prepared_features: bool = False):
        """ Build training data """
        # the_input = np.array(the_input) / 100
        # the_input = x3/np.max(the_input)

        mid_features = [read_audio(audio, sample_rate=self.sample_rate, mono=self.mono) for audio in audios]

        the_input = self.preprocess(mid_features, prepared_features, augmentation)

        the_labels = np.array(the_labels)

        label_len = [len(trans) for trans in transcripts]  # length of each transcription
        label_lengths = np.array(label_len).reshape(-1, 1)  # reshape to 1d

        input_lengths = np.ones((the_labels.shape[0], 1)) * the_labels.shape[1]
        for i in range(the_input.shape[0]):
            input_lengths[i] = the_labels.shape[1]  # num of features from labels

        return {
            'the_input': the_input,
            'the_labels': the_labels,
            'input_length': np.asarray(input_lengths),
            'label_length': np.asarray(label_lengths)
        }

    def save(self, directory: str):
        """ Save each component of the CTC pipeline. """
        # self._model.save(os.path.join(directory, 'model.h5'))
        self._model.save_weights(os.path.join(directory, 'model_weights.h5'))
        save_data(self._alphabet, os.path.join(directory, 'alphabet.bin'))
        save_data(self._decoder, os.path.join(directory, 'decoder.bin'))
        save_data(self._features_extractor,
                  os.path.join(directory, 'feature_extractor.bin'))

    # def load(self, directory: str):
    #     """ Load each component of the CTC pipeline. """
    #     # model = keras.models.load_model(os.path.join(directory, 'model.h5'),
    #     #                                 custom_objects={'clipped_relu': cls.clipped_relu})
    #     self._model.load_weights(os.path.join(directory, 'model_weights.h5'))
    #     self._alphabet = load_data(os.path.join(directory, 'alphabet.bin'))
    #     self._decoder = load_data(os.path.join(directory, 'decoder.bin'))
    #     self._features_extractor = load_data(
    #         os.path.join(directory, 'feature_extractor.bin'))

    @staticmethod
    def distribute_model(model: keras.Model) -> keras.Model:
        """ Replicates a model on different GPUs. """
        try:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            with strategy.scope():
                dist_model = model
            print("Training using multiple GPUs")
            logger.info("Training using multiple GPUs")
        except ValueError:
            dist_model = model
            logger.info("Training using single GPU or CPU")
        return dist_model
