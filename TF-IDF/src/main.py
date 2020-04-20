import os
from os.path import isfile, isdir
from os import listdir

from utils import (
    read_yaml,
    collect_data_from,
    generate_vocabulary,
    get_tf_idf
)

configs = read_yaml('configs.yml')

path = configs['dataroot']
dirs = [os.path.join(path, dir_name) for dir_name in listdir(path) if not isfile(os.path.join(path, dir_name))]
train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
list_newsgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
list_newsgroups.sort()


with open(os.path.join(configs['dataroot'], configs['stop_words'])) as f:
    stop_words = f.read().splitlines()


train_data = collect_data_from(parent_dir = train_dir, newsgroup_list=list_newsgroups, stop_words=stop_words)
test_data = collect_data_from(parent_dir = test_dir, newsgroup_list=list_newsgroups, stop_words=stop_words) 
full_data = train_data + test_data
with open(os.path.join(configs['dataroot'], configs['processed_train']), 'w') as f:
    f.write('\n'.join(train_data))
with open(os.path.join(configs['dataroot'], configs['processed_test']), 'w') as f:
    f.write('\n'.join(test_data))
with open(os.path.join(configs['dataroot'], configs['processed_full']), 'w') as f:
    f.write('\n'.join(full_data))


generate_vocabulary(os.path.join(configs['dataroot'], configs['processed_full']),
                    os.path.join(configs['dataroot'], configs['words_idfs']))


get_tf_idf(os.path.join(configs['dataroot'], configs['processed_train']),
           os.path.join(configs['dataroot'], configs['words_idfs']),
           os.path.join(configs['dataroot'], configs['train_tf_idf']))
get_tf_idf(os.path.join(configs['dataroot'], configs['processed_test']),
           os.path.join(configs['dataroot'], configs['words_idfs']),
           os.path.join(configs['dataroot'], configs['test_tf_idf']))
