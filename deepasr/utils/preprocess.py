from features.filter_banks import FilterBanks
from features.spectrogram import Spectrogram
import numpy as np


def get_features(audios, feature_type='fbank', feature_num=160):
    if feature_type == 'fbank':
        feature_map = FilterBanks(features_num=feature_num)
        return np.asanyarray([feature_map.get_features(audio) for audio in audios])
    else:
        feature_map = Spectrogram(
            features_num=160,
            samplerate=16000,
            winlen=0.02,
            winstep=0.01,
            winfunc=np.hanning
        )
        return np.asanyarray([feature_map.get_features(audio) for audio in audios])
