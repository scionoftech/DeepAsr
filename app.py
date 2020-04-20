import numpy as np
import pandas as pd
import tensorflow as tf
import deepasr as asr


def get_config(features, multi_gpu):
    alphabet_en = asr.vocab.Alphabet(lang='en')

    features_extractor = asr.features.preprocess(feature_type=features, features_num=161,
                                                 samplerate=16000,
                                                 winlen=0.02,
                                                 winstep=0.01,
                                                 winfunc=np.hanning)

    model = asr.model.get_deepspeech2(
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
    # history = pipeline.fit_iter(train_data, batch_size=batch_size, epochs=epochs, iter_num=1000)
    history = pipeline.fit_generator(train_data, batch_size=batch_size, epochs=epochs)
    pipeline.save('./checkpoints')
    print("Truth:", test_data['transcript'].to_list()[0])
    print("Prediction", pipeline.predict(test_data['path'].to_list()[0]))
    return history


def test_model(test_data):
    pipeline = asr.pipeline.load('checkpoints')
    print("Truth:", test_data['transcripts'].to_list()[0])
    print("Prediction", pipeline.predict(test_data['path'].to_list()[0]))


if __name__ == "__main__":
    train = pd.read_csv('train_data.csv')
    test = pd.read_csv('test_data.csv')
    run(train, test, features='fbank', batch_size=32, epochs=100, multi_gpu=False)
    # test_model(test)
