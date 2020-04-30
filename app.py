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
    model = asr.model.get_deepasrnetwork1(
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
    # CTCPipeline
    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=alphabet_en, features_extractor=features_extractor, model=model, optimizer=optimizer, decoder=decoder,
        sample_rate=16000, mono=True, multi_gpu=multi_gpu
    )
    return pipeline


def run():

    train_data = pd.read_csv('train_data.csv')

    pipeline = get_config(feature_type = 'fbank', multi_gpu=False)

    # train asr model
    history = pipeline.fit(train_dataset=train_data, batch_size=128, epochs=500)
    # history = pipeline.fit_generator(train_dataset = train_data, batch_size=32, epochs=500)

    pipeline.save('./checkpoints')

    return history


def test_model(test_data):
    test_data = pd.read_csv('test_data.csv')
    pipeline = asr.pipeline.load('checkpoints')
    print("Truth:", test_data['transcripts'].to_list()[0])
    print("Prediction", pipeline.predict(test_data['path'].to_list()[0]))


if __name__ == "__main__":
    run()
    # test_model(test)
