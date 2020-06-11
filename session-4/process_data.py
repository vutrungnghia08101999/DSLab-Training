import argparse
from collections import defaultdict
import logging
import os
import re
from tqdm import tqdm

logging.basicConfig(filename='process_data_logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/media/vutrungnghia/New Volume/DSLab/dataset')
parser.add_argument('--w2v_path', type=str, default='/media/vutrungnghia/New Volume/DSLab/dataset/w2v')
args = parser.parse_args()

logging.info(args._get_kwargs())

def get_data_and_vocab(root: str):
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        logging.info(f'Processing: {parent_path}')
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = os.path.join(parent_path, newsgroup)
            files = [(filename, os.path.join(dir_path, filename)) for filename in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, filename))]
            files.sort()
            label = group_id
            logging.info(f'Processed newsgroup: {newsgroup} - label: {label} - n_files: {len(files)}')
            for filename, filepath in files:
                with open(filepath, 'rb') as f:
                    text = f.read().decode('UTF-8', errors='ignore').lower()
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data
    word_count = defaultdict(int)
    train_path = os.path.join(root, '20news-bydate-train')
    test_path = os.path.join(root, '20news-bydate-test')
    newsgroup_list = os.listdir(train_path)
    newsgroup_list.sort()  # the same for train and test set
    train_data = collect_data_from(
        parent_path=train_path,
        newsgroup_list=newsgroup_list,
        word_count=word_count
    )

    train_vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
    logging.info(f'train vocab_size: {len(train_vocab)}/{len(word_count)}')

    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroup_list
    )
    return train_data, test_data, train_vocab




data_path = os.path.join(args.w2v_path, 'train.txt')
vocab_path = os.path.join(args.w2v_path, 'vocab.txt')
def encode_data(data_path: str, vocab_path: str, MAX_LENGTHS=500, unkown_word=0, padding_word=1) -> list:
    """encode text (list of words) to list of id in vocab dictionary
    Return:
        list
    """
    vocab_list = open(vocab_path).read().splitlines()
    vocab = dict([(word, wordID + 2) for wordID, word in enumerate(vocab_list)])

    data = open(data_path).read().splitlines()
    documents = [(news.split('<fff>')[0], news.split('<fff>')[1], news.split('<fff>')[2]) for news in data]
    encoded_data = []

    for document in tqdm(documents):
        label, doc_id, text = document
        words = text.split()[: MAX_LENGTHS]
        sentence_length = len(words)  # min(500, len(text.split()))
        encode_text = []
        for word in words:
            if word in vocab:
                encode_text.append(str(vocab[word]))
            else:
                encode_text.append(str(unkown_word))
        if len(encode_text) < MAX_LENGTHS:
            n = MAX_LENGTHS - len(encode_text)
            encode_text = encode_text + [str(padding_word)] * n
        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' + str(sentence_length) + '<fff>' + ' '.join(encode_text))

    return encoded_data

logging.info('Create vocabulary with freq > 10')
train_data, test_data, vocab = get_data_and_vocab(args.dataroot)
with open(os.path.join(args.w2v_path, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_data))
with open(os.path.join(args.w2v_path, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_data))
with open(os.path.join(args.w2v_path, 'vocab.txt'), 'w') as f:
    f.write('\n'.join(vocab))

logging.info('Encode train and test data with vocab, 0 for unkown and 1 for padding, size = 500')
train_encoded = encode_data(
    data_path=os.path.join(args.w2v_path, 'train.txt'),
    vocab_path=os.path.join(args.w2v_path, 'vocab.txt')
)

test_encoded = encode_data(
    data_path=os.path.join(args.w2v_path, 'test.txt'),
    vocab_path=os.path.join(args.w2v_path, 'vocab.txt')
)

with open(os.path.join(args.w2v_path, 'train_encoded.txt'), 'w') as f:
    f.write('\n'.join(train_encoded))

with open(os.path.join(args.w2v_path, 'test_encoded.txt'), 'w') as f:
    f.write('\n'.join(test_encoded))

logging.info(f'Conpleted')
