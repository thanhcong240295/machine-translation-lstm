import numpy as np


class Encoder:
    def __init__(self, cell, embedding_matrix):
        self.cell = cell
        self.embedding = embedding_matrix
        self.cache = []
        self.grads = None

    def forward(self, index_sequence):
        self.cache = []
        hidden_state = None
        cell_state = None

        for idx in index_sequence:
            x = self.embedding[idx]
            hidden_state, cell_state = self.cell.forward(x, hidden_state, cell_state)

            self.cache.append(self.cell.cache)

        return hidden_state, cell_state

    def backward(self, dh_next, dc_next):
        if not self.cache:
            raise ValueError("No cache found. Call forward() before backward().")

        last_cache = self.cache[-1]
        c_t = last_cache[7]
        H, B = c_t.shape

        if dh_next is None:
            dh_next = np.zeros((H, B))
        if dc_next is None:
            dc_next = np.zeros((H, B))

        grads_sum = {
            "dWf": np.zeros_like(self.cell.Wf),
            "dUf": np.zeros_like(self.cell.Uf),
            "dbf": np.zeros_like(self.cell.bf),
            "dWi": np.zeros_like(self.cell.Wi),
            "dUi": np.zeros_like(self.cell.Ui),
            "dbi": np.zeros_like(self.cell.bi),
            "dWo": np.zeros_like(self.cell.Wo),
            "dUo": np.zeros_like(self.cell.Uo),
            "dbo": np.zeros_like(self.cell.bo),
            "dWc": np.zeros_like(self.cell.Wc),
            "dUc": np.zeros_like(self.cell.Uc),
            "dbc": np.zeros_like(self.cell.bc),
        }

        for cache_t in reversed(self.cache):
            self.cell.cache = cache_t
            grads_t, _, dh_next, dc_next = self.cell.backward(da_next=dh_next, dc_next=dc_next, dy_t=None)

            for k in grads_sum:
                grads_sum[k] += grads_t[k]

        self.grads = grads_sum
        return grads_sum, dh_next, dc_next

    def update(self, lr):
        if self.grads is None:
            return

        self.cell.Wf -= lr * self.grads["dWf"]
        self.cell.Uf -= lr * self.grads["dUf"]
        self.cell.bf -= lr * self.grads["dbf"]

        self.cell.Wi -= lr * self.grads["dWi"]
        self.cell.Ui -= lr * self.grads["dUi"]
        self.cell.bi -= lr * self.grads["dbi"]

        self.cell.Wo -= lr * self.grads["dWo"]
        self.cell.Uo -= lr * self.grads["dUo"]
        self.cell.bo -= lr * self.grads["dbo"]

        self.cell.Wc -= lr * self.grads["dWc"]
        self.cell.Uc -= lr * self.grads["dUc"]
        self.cell.bc -= lr * self.grads["dbc"]
