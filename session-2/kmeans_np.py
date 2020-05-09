import logging
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix

logging.basicConfig(filename='logs_eu.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='/media/vutrungnghia/New Volume/DSLab/datasets/train_tf_idf.txt')
parser.add_argument('--test_path', type=str, default='/media/vutrungnghia/New Volume/DSLab/datasets/test_tf_idf.txt')
parser.add_argument('--word_idfs_path', type=str, default='/media/vutrungnghia/New Volume/DSLab/datasets/words_idfs.txt')
parser.add_argument('--similarity', type=str, required=True)
args = parser.parse_args()


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self.r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []

    def reset_members(self):
        self._members = []
    
    def add_member(self, member: Member):
        self._members.append(member)


class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(num_clusters)]
        self._E = []
        self._S = 0
    
    def load_data(self, data_path: str, word_idfs_path: str):
        """Load train data and store each news as (vector tfidf, label and doc_id)

        Arguments:
            data_path {str} -- [path to train data]
            word_idfs_path {str} -- [path to word_idfs]
        """
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
        
        self._data = []
        self._label_count = defaultdict(int)
        for d in tqdm(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_rd=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d, label, doc_id))
    
    def run(self, similarity_type: str):
        for cluster in self._clusters:
            random_int = np.random.randint(len(self._data))
            cluster._centroid =  self._data[random_int].r_d
        
        self._iteration = 1
        last_purity = -1
        last_NMI = -1
        while True:
            logging.info('Reset all clusters to empty')
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            logging.info('Choose cluster for each member in data')
            for member in tqdm(self._data):
                max_S = self.select_cluster_for(member, similarity_type)
                self._new_S += max_S
            for cluster in self._clusters:
                self.update_centroid_for(cluster)

            self._iteration += 1
            logging.info(f"Completed iteration {self._iteration}")
            self.print()
            purity = self.compute_purity()
            NMI = self.compute_NMI()
            logging.info(f'Purity metric: {purity}')
            logging.info(f'NMI metric: {NMI}')

            if purity - last_purity < 0.001 or NMI - last_NMI < 0.001:
                break
            last_purity = purity
            last_NMI = NMI

    def select_cluster_for(self, member: Member, similarity_type) -> float:
        """Add member to cluster in self._cluster and return correspond similarity

        Arguments:
            member {Member}

        Returns:
            [float] -- similarity(member.rd, best_fit_cluster._centroid)
        """
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            if similarity_type == 'COSIN':
                similarity = self.compute_cosin_similarity(member.r_d, cluster._centroid)
            elif similarity_type == 'EUCLIDEAN':
                similarity = self.compute_euclidean_similarity(member.r_d , cluster._centroid)
            else:
                raise RuntimeError(f'{similarity} is invalid')

            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity

    def update_centroid_for(self, cluster: Cluster):
        """Average all members in cluster and L2Norm = 1

        Arguments:
            cluster {Cluster} -- previous centroid and list of new members
        """
        if len(cluster._members) == 0:
            random_int = np.random.randint(len(self._data))
            cluster._centroid = self._data[random_int].r_d
        else:
            member_r_ds = [member.r_d for member in cluster._members]
            aver_r_d = np.mean(member_r_ds, axis=0)
            l2_norm = np.sqrt(np.sum(aver_r_d ** 2))
            new_centroid = aver_r_d/l2_norm  # l2Norm(new_centroid) = 1

            cluster._centroid = new_centroid

    def compute_cosin_similarity(self, vector_a: np.array, vector_b: np.array) -> float:
        # return sum(vector_a * vector_b)
        return csr_matrix(vector_a).multiply(csr_matrix(vector_b)).sum()
    
    def compute_euclidean_similarity(self, vector_a, vector_b) -> float:
        # return 1.0 / (np.linalg.norm(vector_a - vector_b) + 1e-6)
        return 1.0 / (np.sqrt(
            (csr_matrix(vector_a) - csr_matrix(vector_b)).multiply((csr_matrix(vector_a) - csr_matrix(vector_b))).sum()) + 1e-6)

    def print(self):
        for i in range(len(self._clusters)):
            logging.info(f'No.members in cluster {i}: {len(self._clusters[i]._members)}')

    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(self._num_clusters)])
            majority_sum += max_count
        return majority_sum * 1.0 / len(self._data)
    
    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0.0, 0.0, 0.0, len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.0
            H_omega += -wk / N * np.log10(wk/N)
            member_labels = [member._label for member in cluster._members]
            for label in range(self._num_clusters):
                wk_cj = member_labels.count(label) * 1.0
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(self._num_clusters):
            cj = self._label_count[label] * 1.0
            H_C += -cj / N * np.log10(cj/N)
        return I_value * 2.0 / (H_omega + H_C)

for seed in [0, 1, 2, 3]:
    np.random.seed(seed)
    logging.info(f'KMEAN-SEED-{seed}-{args.similarity}-SIMILARITY')
    kmeans = Kmeans(20)
    kmeans.load_data(data_path=args.train_path, word_idfs_path=args.word_idfs_path)
    kmeans.run(args.similarity)
