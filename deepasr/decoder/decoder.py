import abc
# import itertools
from typing import List
import numpy as np
from tensorflow.keras import backend as K


# https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode

class Decoder:

    @abc.abstractmethod
    def __call__(self, batch_logits: np.ndarray, input_length: int) -> List[np.ndarray]:
        pass


class GreedyDecoder:

    def __call__(self, batch_logits: np.ndarray, input_length: int) -> List[np.ndarray]:
        """ Decode the best guess from logits using greedy algorithm. """
        # Choose the class with maximum probability
        # best_candidates = np.argmax(batch_logits, axis=2)
        # Merge repeated chars
        # decoded = [np.array([k for k, _ in itertools.groupby(best_candidate)])
        #            for best_candidate in best_candidates]
        decoded = np.array(
            (K.eval(K.ctc_decode(batch_logits, [input_length], greedy=True)[0][0])).flatten().tolist())
        return [decoded]


class BeamSearchDecoder:

    def __init__(self, beam_width: int, top_paths: int):
        self.beam_width = beam_width
        self.top_paths = top_paths

    def __call__(self, batch_logits: np.ndarray, input_length: int, **kwargs) -> List[
        np.ndarray]:
        """ Decode the best guess from logits using beam search algorithm. """
        decoded = np.array((K.eval(
            K.ctc_decode(batch_logits, [input_length], greedy=False, beam_width=self.beam_width,
                         top_paths=self.top_paths)[0][
                0])).flatten().tolist())
        return [decoded]
