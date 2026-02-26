import numpy as np

from utils.activation import Activation


class LstmCell:
    def __init__(self, input_size, hidden_size, output_size=None):
        self.activation = Activation()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wf = np.random.uniform(-0.01, 0.01, (hidden_size, input_size))
        self.Uf = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        self.bf = np.zeros((hidden_size, 1))

        self.Wi = np.random.uniform(-0.01, 0.01, (hidden_size, input_size))
        self.Ui = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        self.bi = np.zeros((hidden_size, 1))

        self.Wo = np.random.uniform(-0.01, 0.01, (hidden_size, input_size))
        self.Uo = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        self.bo = np.zeros((hidden_size, 1))

        self.Wc = np.random.uniform(-0.01, 0.01, (hidden_size, input_size))
        self.Uc = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        self.bc = np.zeros((hidden_size, 1))

        if output_size is not None:
            self.V = np.random.uniform(-0.01, 0.01, (output_size, hidden_size))
            self.b_out = np.zeros((output_size, 1))
        else:
            self.V = None
            self.b_out = None

        self.cache = None

    def _normalize_input(self, x, expected_size, name):
        x = np.asarray(x)

        if x.ndim == 1:
            if x.shape[0] != expected_size:
                raise ValueError(f"{name} expected shape ({expected_size},) but got {x.shape}")
            return x.reshape(expected_size, 1)

        if x.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D, got {x.ndim}D")

        if x.shape[0] == expected_size:
            return x
        if x.shape[1] == expected_size:
            return x.T

        raise ValueError(
            f"{name} expected shape ({expected_size}, batch) or (batch, {expected_size}) but got {x.shape}"
        )

    def forward(self, x_t, a_prev=None, c_prev=None):
        x_t = self._normalize_input(x_t, self.input_size, "x_t")
        B = x_t.shape[1]

        if a_prev is None:
            a_prev = np.zeros((self.hidden_size, B))
        else:
            a_prev = self._normalize_input(a_prev, self.hidden_size, "a_prev")

        if c_prev is None:
            c_prev = np.zeros((self.hidden_size, B))
        else:
            c_prev = self._normalize_input(c_prev, self.hidden_size, "c_prev")

        if a_prev.shape[1] != B or c_prev.shape[1] != B:
            raise ValueError("Batch size mismatch among x_t, a_prev, c_prev")

        f_t = self.activation.sigmoid(self.Wf @ x_t + self.Uf @ a_prev + self.bf)
        i_t = self.activation.sigmoid(self.Wi @ x_t + self.Ui @ a_prev + self.bi)
        o_t = self.activation.sigmoid(self.Wo @ x_t + self.Uo @ a_prev + self.bo)
        c_tilde = self.activation.tanh(self.Wc @ x_t + self.Uc @ a_prev + self.bc)

        c_t = f_t * c_prev + i_t * c_tilde
        tanh_c = np.tanh(c_t)
        a_t = o_t * tanh_c

        self.cache = (x_t, a_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t, tanh_c, a_t)

        if self.output_size is None:
            return a_t, c_t

        y_t = self.activation.softmax(self.V @ a_t + self.b_out)
        return a_t, c_t, y_t

    def backward(self, da_next, dc_next, dy_t=None):
        if self.cache is None:
            raise ValueError("No cache found. Call forward() before backward().")

        x_t, a_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t, tanh_c, a_t = self.cache

        da_next = self._normalize_input(da_next, self.hidden_size, "da_next")
        dc_next = self._normalize_input(dc_next, self.hidden_size, "dc_next")

        if self.output_size is not None and dy_t is not None:
            dy_t = self._normalize_input(dy_t, self.output_size, "dy_t")

            dV = dy_t @ a_t.T
            db_out = np.sum(dy_t, axis=1, keepdims=True)

            da_next = da_next + (self.V.T @ dy_t)
        else:
            dV = None
            db_out = None

        do = da_next * tanh_c
        dtanh_c = da_next * o_t

        dc = dc_next + dtanh_c * (1.0 - tanh_c**2)

        df = dc * c_prev
        dc_prev = dc * f_t

        di = dc * c_tilde
        dc_tilde = dc * i_t

        df_raw = df * (f_t * (1.0 - f_t))
        di_raw = di * (i_t * (1.0 - i_t))
        do_raw = do * (o_t * (1.0 - o_t))
        dc_tilde_raw = dc_tilde * (1.0 - c_tilde**2)

        dWf = df_raw @ x_t.T
        dUf = df_raw @ a_prev.T
        dbf = np.sum(df_raw, axis=1, keepdims=True)

        dWi = di_raw @ x_t.T
        dUi = di_raw @ a_prev.T
        dbi = np.sum(di_raw, axis=1, keepdims=True)

        dWo = do_raw @ x_t.T
        dUo = do_raw @ a_prev.T
        dbo = np.sum(do_raw, axis=1, keepdims=True)

        dWc = dc_tilde_raw @ x_t.T
        dUc = dc_tilde_raw @ a_prev.T
        dbc = np.sum(dc_tilde_raw, axis=1, keepdims=True)

        da_prev = self.Uf.T @ df_raw + self.Ui.T @ di_raw + self.Uo.T @ do_raw + self.Uc.T @ dc_tilde_raw

        dx = self.Wf.T @ df_raw + self.Wi.T @ di_raw + self.Wo.T @ do_raw + self.Wc.T @ dc_tilde_raw

        grads = {
            "dWf": dWf,
            "dUf": dUf,
            "dbf": dbf,
            "dWi": dWi,
            "dUi": dUi,
            "dbi": dbi,
            "dWo": dWo,
            "dUo": dUo,
            "dbo": dbo,
            "dWc": dWc,
            "dUc": dUc,
            "dbc": dbc,
            "dV": dV,
            "db_out": db_out,
        }

        return grads, dx, da_prev, dc_prev
