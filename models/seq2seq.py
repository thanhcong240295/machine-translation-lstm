import numpy as np

from utils.losses import Losses


class Seq2Seq:
    def __init__(self, encoder, decoder, lr=0.01, sos_idx=0, eos_idx=1):
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src_indices, dec_in_indices):
        context = self.encoder.forward(src_indices)
        outputs = self.decoder.forward(context, dec_in_indices)
        return outputs

    def train_step(self, src_indices, trg_indices):
        if len(src_indices) == 0 or len(trg_indices) < 2:
            return 0.0

        # teacher forcing shift
        dec_in = trg_indices[:-1]
        dec_tg = trg_indices[1:]

        outputs = self.forward(src_indices, dec_in)
        loss, douts = Losses.sequence_nll_with_grads(outputs, dec_tg)

        dec_grads, dh0, dc0 = self.decoder.backward(douts)

        self.encoder.backward(dh0, dc0)

        self.decoder.update(self.lr)
        self.encoder.update(self.lr)

        return loss

    def train(self, dataset, epochs=10):
        for epoch in range(epochs):
            total_loss = 0.0
            n = 0

            for src_indices, trg_indices in dataset:
                loss = self.train_step(src_indices, trg_indices)
                if loss == 0.0 and (len(src_indices) == 0 or len(trg_indices) < 2):
                    continue
                total_loss += loss
                n += 1

            avg = total_loss / max(n, 1)
            print(f"Seq2Seq: Epoch {epoch} Loss {total_loss:.6f} Avg {avg:.6f}")

    def translate(self, src_indices, max_len=50):
        hidden_state, cell_state = self.encoder.forward(src_indices)

        input_token = self.sos_idx
        outputs = []

        for step in range(max_len):
            x = self.decoder.embedding[input_token]
            hidden_state, cell_state, y = self.decoder.cell.forward(x, hidden_state, cell_state)

            y = np.asarray(y).reshape(-1)

            # Greedy decoding
            next_token = int(np.argmax(y))

            if step == 0 and next_token == self.eos_idx:
                next_token = int(np.argsort(y)[-2])

            if next_token == self.eos_idx:
                break

            outputs.append(next_token)
            input_token = next_token

        return outputs
