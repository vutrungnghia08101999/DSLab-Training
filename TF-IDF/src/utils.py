from collections import defaultdict
import numpy as np
from nltk.stem.porter import PorterStemmer
import os
from os.path import isfile
import re
from tqdm import tqdm
import yaml


def read_yaml(filename: str) -> dict:
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def collect_data_from(parent_dir: str, newsgroup_list: list, stop_words: list) -> list:
    data = []
    stemmer = PorterStemmer()
    for group_id, newsgroup in tqdm(enumerate(newsgroup_list)):
        label = group_id
        dir_path = parent_dir + '/' + newsgroup + '/'
        files = [(filename, dir_path + filename) for filename in os.listdir(dir_path) if isfile(dir_path + filename)]
        files.sort()
        for filename, filepath in files:
            with open(filepath, encoding='ISO-8859-1') as f:
                text = f.read().lower()
                words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                content = ' '.join(words)
                assert  len(content.splitlines()) == 1
                data.append(str(label) + '<fff>' + filename + '<fff>' + content)
    return data


def generate_vocabulary(processed_data_path: str, words_idfs_path: str) -> None:
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1/df)

    with open(processed_data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)

    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1

    words_idfs = [(word, compute_idf(df, corpus_size))
        for word, df in zip(doc_count.keys(), doc_count.values())
        if df > 10 and not word.isdigit()]
    words_idfs.sort(key=lambda X: -X[1])
    print(f'Vocabulary size {len(words_idfs)}')
    with open(words_idfs_path, 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))


def get_tf_idf(processed_data_path: str, words_idfs_path: str, tf_idf_path: str) -> None:
    with open(words_idfs_path) as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                     for line in f.read().splitlines()]
        word_ID = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)
    with open(processed_data_path) as f:
        documents = [(int(line.split('<fff>')[0]),
                     int(line.split('<fff>')[1]),
                     line.split('<fff>')[2])
                     for line in f.read().splitlines()]
    data_tf_idf = []
    for document in tqdm(documents):
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tfidfs = []
        sum_squares = 0.0

        for word in word_set:
                term_freq = words.count(word)
                tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
                words_tfidfs.append((word_ID[word], tf_idf_value))
                sum_squares += tf_idf_value ** 2

        words_tfidfs_normalized = [str(index) + ':' + str(tf_idf_value / np.sqrt(sum_squares))
                                    for index, tf_idf_value in words_tfidfs]

        sparse_rep = ' '.join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    with open(tf_idf_path, 'w') as f:
        f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep for (label, doc_id, sparse_rep) in data_tf_idf]))
