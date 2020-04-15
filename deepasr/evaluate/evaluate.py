from typing import List, Iterable, Tuple, Union
from collections import namedtuple
import pandas as pd
from . import distance
# from .. import dataset
from .. import pipeline

Metric = namedtuple('Metric', ['transcript', 'prediction', 'wer', 'cer'])


def calculate_error_rates(ctc_pipeline: pipeline.Pipeline,
                          data: pd.DataFrame,
                          return_metrics: bool = False
                          ) -> Union[Tuple[float, float], pd.DataFrame]:
    """ Calculate base metrics: WER and CER. """
    metrics = []
    for audio, transcript in zip(data['path'].values, data['transcripts'].values):
        prediction = ctc_pipeline.predict(audio)
        batch_metrics = get_metrics(sources=prediction,
                                    destinations=[transcript])
        metrics.extend(batch_metrics)
    metrics = pd.DataFrame(metrics)
    return metrics if return_metrics else (metrics.wer.mean(), metrics.cer.mean())


def get_metrics(sources: List[str],
                destinations: List[str]) -> Iterable[Metric]:
    """ Calculate base metrics in one batch: WER and CER. """
    for source, destination in zip(sources, destinations):
        wer_distance, *_ = distance.edit_distance(source.split(),
                                                  destination.split())
        wer = wer_distance / len(destination.split())

        cer_distance, *_ = distance.edit_distance(list(source),
                                                  list(destination))
        cer = cer_distance / len(destination)
        yield Metric(destination, source, wer, cer)


def get_cer(source: str, destination: str) -> float:
    cer_distance, *_ = distance.edit_distance(list(source),
                                              list(destination))
    cer = cer_distance / len(destination)

    return cer


def get_wer(source: str, destination: str) -> float:
    wer_distance, *_ = distance.edit_distance(source.split(),
                                              destination.split())
    wer = wer_distance / len(destination.split())

    return wer
