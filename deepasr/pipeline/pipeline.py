import abc
from typing import List
import numpy as np
import pandas as pd
from tensorflow import keras
import sys

sys.path.append("..")
from deepasr.decoder import Decoder
from deepasr.features import FeaturesExtractor
from deepasr.vocab import Alphabet


class Pipeline:

    @property
    @abc.abstractmethod
    def alphabet(self) -> Alphabet:
        pass

    @property
    @abc.abstractmethod
    def features_extractor(self) -> FeaturesExtractor:
        pass

    @property
    @abc.abstractmethod
    def model(self) -> keras.Model:
        pass

    @property
    @abc.abstractmethod
    def decoder(self) -> Decoder:
        pass

    @abc.abstractmethod
    def fit(self,
            train_dataset: pd.DataFrame,
            val_dataset: pd.DataFrame,
            prepared_features=False,
            **kwargs) -> keras.callbacks.History:
        pass

    @abc.abstractmethod
    def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
        pass

    @abc.abstractmethod
    def save(self, directory: str):
        pass
