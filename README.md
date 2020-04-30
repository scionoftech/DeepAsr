# DeepAsr
DeepAsr is an open-source & Keras (Tensorflow) implementation of end-to-end Automatic Speech Recognition (ASR) engine and it supports multiple Speech Recognition architectures.

Supported Asr Architectures:
- Baidu's Deep Speech 2
- DeepAsrNetwork1

**Using DeepAsr you can**:
- perform speech-to-text using pre-trained models
- tune pre-trained models to your needs
- create new models on your own 

**DeepAsr key features**:
- **Multi GPU support**: You can do much more like distribute the training using the [Strategy](https://www.tensorflow.org/guide/distributed_training), or experiment with [mixed precision](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental/Policy) policy.
- **CuDNN support**: Model using [CuDNNLSTM](https://keras.io/layers/recurrent/) implementation by NVIDIA Developers. CPU devices is also supported.
- **DataGenerator**: The feature extraction during model training for large the data.

## Installation
You can use pip:
```bash
pip install deepasr
```

## Getting started
The speech recognition is a tough task. You don't need to know all details to use one of the pretrained models.
However it's worth to understand conceptional crucial components:
- **Input**: Audio files (WAV or FLAC) with mono 16-bit 16 kHz (up to 5 seconds)
- **FeaturesExtractor**: Convert audio files using MFCC Features or Spectrogram
- **Model**: CTC model defined in [**Keras**](https://keras.io/) (references: [[1]](https://arxiv.org/abs/1412.5567), [[2]](https://arxiv.org/abs/1512.02595))
- **Decoder**: Greedy or BeamSearch algorithms with the language model support decode a sequence of probabilities using Alphabet
- **DataGenerator**: Stream data to the model via generator
- **Callbacks**: Set of functions monitoring the training

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import deepasr as asr

# get CTCPipeline
def get_config(feature_type: str = 'spectrogram', multi_gpu: bool = False):
    # audio feature extractor
    features_extractor = asr.features.preprocess(feature_type=feature_type, features_num=161,
                                                 samplerate=16000,
                                                 winlen=0.02,
                                                 winstep=0.025,
                                                 winfunc=np.hanning)

    # input label encoder
    alphabet_en = asr.vocab.Alphabet(lang='en')
    # training model
    model = asr.model.get_deepspeech2(
        input_dim=161,
        output_dim=29,
        is_mixed_precision=True
    )
    # model optimizer
    optimizer = tf.keras.optimizers.Adam(
        lr=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    # output label deocder
    decoder = asr.decoder.GreedyDecoder()
    # decoder = asr.decoder.BeamSearchDecoder(beam_width=100, top_paths=1)
    # CTCPipeline
    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=alphabet_en, features_extractor=features_extractor, model=model, optimizer=optimizer, decoder=decoder,
        sample_rate=16000, mono=True, multi_gpu=multi_gpu
    )
    return pipeline


train_data = pd.read_csv('train_data.csv')

pipeline = get_config(feature_type = 'fbank', multi_gpu=False)

# train asr model
history = pipeline.fit(train_dataset=train_data, batch_size=128, epochs=500)
# history = pipeline.fit_generator(train_dataset = train_data, batch_size=32, epochs=500)

pipeline.save('./checkpoint')
```

Loaded pre-trained model has all components. The prediction can be invoked just by calling pipline.predict().

```python
import pandas as pd
import deepasr as asr
import numpy as np
test_data = pd.read_csv('test_data.csv')

# get testing audio and transcript from dataset
index = np.random.randint(test_data.shape[0])
data = test_data.iloc[index]
test_file = data[0]
test_transcript = data[1]
# Test Audio file
print("Audio File:",test_file)
# Test Transcript
print("Audio Transcript:", test_transcript)
print("Transcript length:",len(test_transcript))

pipeline = asr.pipeline.load('./checkpoint')
print("Prediction", pipeline.predict(test_file))
```

#### References

The fundamental repositories:
- Baidu - [DeepSpeech2 - A PaddlePaddle implementation of DeepSpeech2 architecture for ASR](https://github.com/PaddlePaddle/DeepSpeech)
- NVIDIA - [Toolkit for efficient experimentation with Speech Recognition, Text2Speech and NLP](https://nvidia.github.io/OpenSeq2Seq)
- TensorFlow - [The implementation of DeepSpeech2 model](https://github.com/tensorflow/models/tree/master/research/deep_speech)
- Mozilla - [DeepSpeech - A TensorFlow implementation of Baidu's DeepSpeech architecture](https://github.com/mozilla/DeepSpeech) 
- Espnet - [End-to-End Speech Processing Toolkit](https://github.com/espnet/espnet)
- Automatic Speech Recognition - [Distill the Automatic Speech Recognition research](https://github.com/rolczynski/Automatic-Speech-Recognition)
- Python Speech Features - [Speech features for ASR including MFCCs and filterbank energies](https://github.com/jameslyons/python_speech_features)