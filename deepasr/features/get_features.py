from .filter_banks import FilterBanks
from .spectrogram import Spectrogram


def preprocess(feature_type: str = 'fbank', features_num: int = 161,
               samplerate: int = 16000,
               winlen: float = 0.02,
               winstep: float = 0.01,
               winfunc=None,
               is_standardization=True,
               pad_audio_to: int = 0):
    ''' This method extracts the audio features based on fbank or spectrogram '''
    if feature_type == 'fbank':
        features_extractor = FilterBanks(features_num=features_num, samplerate=samplerate, winlen=winlen,
                                         winstep=winstep, winfunc=winfunc,
                                         is_standardization=is_standardization)
        return features_extractor
    elif feature_type == 'spectrogram':
        features_extractor = Spectrogram(
            features_num=features_num,
            samplerate=samplerate,
            winlen=winlen,
            winstep=winstep,
            winfunc=winfunc,
            pad_audio_to=pad_audio_to
        )
        return features_extractor
