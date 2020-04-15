# DeepAsr
DeepAsr is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine. 

DeepAsr will provide multiple Speech Recognition Deep Neural Network architectures, Currenly it provides Baidu's Deep Speech 2 using Keras (Tensorflow).

**Using DeepAsr you can**:
- perform speech-to-text using pre-trained models
- tune pre-trained models to your needs
- create new models on your own 

**DeepAsr key features**:
- **Multi GPU support**: You can do much more like distribute the training using the [Strategy](https://www.tensorflow.org/guide/distributed_training), or experiment with [mixed precision](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental/Policy) policy.
- **CuDNN support**: Model using [CuDNNLSTM](https://keras.io/layers/recurrent/) implementation by NVIDIA Developers. CPU devices is also supported.
- **DataGenerator**: The feature extraction (on CPU) can be parallel to model training (on GPU).


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import deepasr as asr

def get_config(features, multi_gpu):
    alphabet_en = asr.vocab.Alphabet(lang='en')
    if features == 'fbank':
        features_extractor = asr.features.FilterBanks(features_num=161,
                                                      winlen=0.02,
                                                      winstep=0.01,
                                                      winfunc=np.hanning)
    else:
        features_extractor = asr.features.Spectrogram(
            features_num=161,
            samplerate=16000,
            winlen=0.02,
            winstep=0.01,
            winfunc=np.hanning
        )
    model = asr.model.get_deepspeech2_v1(
        input_dim=161,
        output_dim=29,
        is_mixed_precision=True
        )
    optimizer = tf.keras.optimizers.Adam(
        lr=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
        )
    decoder = asr.decoder.GreedyDecoder()

    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=alphabet_en, features_extractor=features_extractor, model=model, optimizer=optimizer, decoder=decoder,
        sample_rate=16000, mono=True, multi_gpu=multi_gpu
    )
    return pipeline

def run(train_data, test_data, features='fbank', batch_size=32, epochs=10, multi_gpu=True):
    pipeline = get_config(features, multi_gpu)
    history = pipeline.fit_generator(train_data, batch_size=batch_size, epochs=epochs)
    pipeline.save('./checkpoints')
    print("Truth:", test_data['transcripts'].to_list()[0])
    print("Prediction", pipeline.predict(test_data['path'].to_list()[0]))
    return history

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
run(train, test, features='fbank', batch_size=32, epochs=100, multi_gpu=True)
```

## Installation
You can use pip:
```bash
pip install deepspeechasr
```

## Getting started
The speech recognition is a tough task. You don't need to know all details to use one of the pretrained models.
However it's worth to understand conceptional crucial components:
- **Input**: WAVE files with mono 16-bit 16 kHz (up to 5 seconds)
- **FeaturesExtractor**: Convert audio files using MFCC Features or Spectrogram
- **Model**: CTC model defined in [**Keras**](https://keras.io/) (references: [[1]](https://arxiv.org/abs/1412.5567), [[2]](https://arxiv.org/abs/1512.02595))
- **Decoder**: Greedy algorithm with the language model support decode a sequence of probabilities using Alphabet
- **DataGenerator**: Stream data to the model via generator
- **Callbacks**: Set of functions monitoring the training

Loaded pre-trained model has all components. The prediction can be invoked just by calling pipline.predict().

```python
import pandas as pd
import deepasr as asr
pipeline = asr.pipeline.get_pipeline.load('./checkpoints')
test_data = pd.read_csv('test_data.csv')
print("Truth:", test_data['transcripts'].to_list()[0])
print("Prediction", pipeline.predict(test_data['path'].to_list()[0]))
```

#### References

The fundamental repositories:
- Baidu - [DeepSpeech2 - A PaddlePaddle implementation of DeepSpeech2 architecture for ASR](https://github.com/PaddlePaddle/DeepSpeech)
- NVIDIA - [Toolkit for efficient experimentation with Speech Recognition, Text2Speech and NLP](https://nvidia.github.io/OpenSeq2Seq)
- TensorFlow - [The implementation of DeepSpeech2 model](https://github.com/tensorflow/models/tree/master/research/deep_speech)
- Mozilla - [DeepSpeech - A TensorFlow implementation of Baidu's DeepSpeech architecture](https://github.com/mozilla/DeepSpeech) 
- Espnet - [End-to-End Speech Processing Toolkit](https://github.com/espnet/espnet)
- Automatic Speech Recognition - [Distill the Automatic Speech Recognition research](https://github.com/rolczynski/Automatic-Speech-Recognition)