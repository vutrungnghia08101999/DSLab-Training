import numpy as np
import random

MAX_DOC_LENGTH = 500
NUM_CLASSES = 20

class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._labels = []
        self._sentence_lengths = []

        for data_id, line in enumerate(d_lines):
            if len(line) > 1:
                features = line.split('<fff>')
                label = int(features[0])
                doc_id = int(features[1])
                length = int(features[2])
                tokens = features[3].split()
                vector = [int(token) for token in tokens]
                self._data.append(vector)
                self._labels.append(label)
                self._sentence_lengths.append(length)
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sentence_lengths = np.array(self._sentence_lengths)
        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1
        if end + self._batch_size > len(self._data):
            start = len(self._data) - self._batch_size
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0

            indices = [i for i in range(len(self._data))]
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels, self._sentence_lengths = self._data[indices], self._labels[indices], self._sentence_lengths[indices]
        return self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end]
