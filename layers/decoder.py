import numpy as np


class Decoder:
    def __init__(self, cell, embedding_matrix):
        self.cell = cell
        self.embedding = embedding_matrix

        self.cache = []
        self.grads = None

    def forward(self, context, target_indices):
        self.cache = []
        outputs = []

        if not isinstance(context, tuple) or len(context) != 2:
            raise ValueError("LSTM decoder expects context = (hidden_state, cell_state).")

        hidden_state, cell_state = context

        for idx in target_indices:
            x = self.embedding[idx]

            hidden_state, cell_state, y = self.cell.forward(x, hidden_state, cell_state)

            y_out = np.asarray(y).reshape(-1)
            outputs.append(y_out)

            self.cache.append(self.cell.cache)

        return outputs

    def backward(self, douts):
        if not self.cache:
            raise ValueError("No cache found. Call forward() before backward().")

        if len(douts) != len(self.cache):
            raise ValueError(f"douts len {len(douts)} must match timesteps {len(self.cache)}")

        last_cache = self.cache[-1]
        c_t = last_cache[7]
        H, B = c_t.shape

        dh_next = np.zeros((H, B))
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
            "dV": np.zeros_like(self.cell.V) if self.cell.V is not None else None,
            "db_out": np.zeros_like(self.cell.b_out) if getattr(self.cell, "b_out", None) is not None else None,
        }

        for cache_t, dy_t in zip(reversed(self.cache), reversed(douts)):
            self.cell.cache = cache_t

            dy_t = np.asarray(dy_t)
            if dy_t.ndim == 1:
                dy_t = dy_t.reshape(-1, 1)

            grads_t, _, dh_next, dc_next = self.cell.backward(da_next=dh_next, dc_next=dc_next, dy_t=dy_t)

            for k, v in grads_t.items():
                if v is None:
                    continue
                if k == "db_out":
                    if grads_sum["db_out"] is not None:
                        grads_sum["db_out"] += v
                elif k == "dV":
                    if grads_sum["dV"] is not None:
                        grads_sum["dV"] += v
                else:
                    grads_sum[k] += v

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

        if self.cell.V is not None and self.grads["dV"] is not None:
            self.cell.V -= lr * self.grads["dV"]

        if getattr(self.cell, "b_out", None) is not None and self.grads["db_out"] is not None:
            self.cell.b_out -= lr * self.grads["db_out"]
