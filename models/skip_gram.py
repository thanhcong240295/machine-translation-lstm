import numpy as np

from utils.activation import Activation
from utils.losses import Losses


class SkipGram:
    def __init__(self, embedding_dim=100, window_size=2, lr=0.01):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = lr
        self.vocab = None
        self.vocab_size = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.embedding = None
        self.W = None
        self.W_out = None

    def train(self, corpus, epochs=10, batch_size=256):
        self._build_vocabulary(corpus)
        pairs = self.generate_pairs(corpus)
        self._initialize_weights()

        if batch_size is None or batch_size <= 0:
            batch_size = len(pairs)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            loss = 0
            pair_index = 0
            total_pairs = len(pairs)
            total_batches = max((total_pairs + batch_size - 1) // batch_size, 1)
            batch_idx = 0

            for batch_start in range(0, len(pairs), batch_size):
                batch_idx += 1
                self._print_progress(batch_idx, total_batches)
                batch = pairs[batch_start : batch_start + batch_size]
                dW_out = np.zeros_like(self.W_out)
                dW = np.zeros_like(self.W)

                for center, context in batch:
                    pair_index += 1
                    h = self.W[center]
                    u = np.dot(h, self.W_out)
                    y_pred = Activation.softmax(u)
                    loss += Losses.negative_log_likelihood(y_pred, context)
                    e = y_pred.copy()
                    e[context] -= 1

                    dW_out += np.outer(h, e)
                    dW[center] += np.dot(self.W_out, e)

                self.W_out -= self.lr * (dW_out / max(len(batch), 1))
                self.W -= self.lr * (dW / max(len(batch), 1))

            print(f"\r{' ' * 60}", end="\r")
            print(f"Skip-gram: Epoch {epoch + 1}/{epochs}, Loss {loss}")

    def get_embedding(self, word):
        return self.W[self.word_to_idx[word]]

    def get_embedding_matrix(self):
        return self.W

    def generate_pairs(self, corpus):
        return self._generate_pairs(corpus)

    def _initialize_weights(self):
        self.W = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01

    def _build_vocabulary(self, corpus) -> None:
        words = [w for sentence in corpus for w in sentence.split()]
        self.vocab = list(set(words))
        self.vocab_size = len(self.vocab)

        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

        self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01

    def _generate_pairs(self, corpus):
        pairs = []

        for sentence in corpus:
            tokens = sentence.split()
            indices = [self.word_to_idx[w] for w in tokens]

            for i, center in enumerate(indices):
                for j in range(-self.window_size, self.window_size + 1):
                    if j == 0:
                        continue
                    if 0 <= i + j < len(indices):
                        context = indices[i + j]
                        pairs.append((center, context))

        return pairs

    @staticmethod
    def _print_progress(current, total, bar_len=30):
        filled = int(bar_len * current / max(total, 1))
        bar = "█" * filled + "-" * (bar_len - filled)
        percent = (current / max(total, 1)) * 100
        print(f"\r[{bar}] {percent:5.1f}%", end="")
