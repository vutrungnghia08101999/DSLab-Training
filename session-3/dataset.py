from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset


class TFIDFDataset(Dataset):
    def __init__(self, data_path: str, word_idfs_path: str, n_classes=20):
        self.X, self.Y = load_data(data_path, word_idfs_path)
        self.n_classes = n_classes

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        label = np.zeros(self.n_classes)
        X = self.X[index]
        label[self.Y[index]] = 1
        return X, label  # X.shape = 11314, Y.shape = 20

def load_data(data_path, word_idfs_path):
    def sparse_to_dense(sparse_rd, vocab_size):
        r_d = [0.0] * vocab_size
        indices_tfidfs = sparse_rd.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()

    with open(word_idfs_path) as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    for d in tqdm(d_lines):
        features = d.split('<fff>')
        label, _ = int(features[0]), int(features[1])
        labels.append(label)
        r_d = sparse_to_dense(sparse_rd=features[2], vocab_size=vocab_size)
        data.append(r_d.reshape(1, r_d.shape[0]))

    data = np.concatenate(data, axis=0)
    return data, np.array(labels)  # X, Y
