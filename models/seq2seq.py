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

    def evaluate_loss(self, dataset):
        total_loss = 0.0
        n = 0

        for src_indices, trg_indices in dataset:
            if len(src_indices) == 0 or len(trg_indices) < 2:
                continue

            dec_in = trg_indices[:-1]
            dec_tg = trg_indices[1:]

            outputs = self.forward(src_indices, dec_in)
            loss, _ = Losses.sequence_nll_with_grads(outputs, dec_tg)
            total_loss += loss
            n += 1

        return total_loss / max(n, 1)

    def train(self, dataset, epochs=10, batch_size=32, val_dataset=None):
        if batch_size is None or batch_size <= 0:
            batch_size = len(dataset)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            n = 0
            total_batches = max((len(dataset) + batch_size - 1) // batch_size, 1)
            batch_idx = 0

            for batch_start in range(0, len(dataset), batch_size):
                batch_idx += 1
                self._print_progress(batch_idx, total_batches)
                batch = dataset[batch_start : batch_start + batch_size]

                for src_indices, trg_indices in batch:
                    loss = self.train_step(src_indices, trg_indices)
                    if loss == 0.0 and (len(src_indices) == 0 or len(trg_indices) < 2):
                        continue
                    total_loss += loss
                    n += 1

            avg = total_loss / max(n, 1)
            history["train_loss"].append(avg)

            if val_dataset is not None:
                val_loss = self.evaluate_loss(val_dataset)
                history["val_loss"].append(val_loss)
                print(f"\r{' ' * 60}", end="\r")
                print(f"Seq2Seq: Epoch {epoch + 1}/{epochs} " f"Loss {total_loss:.6f} Avg {avg:.6f} Val {val_loss:.6f}")
            else:
                print(f"\r{' ' * 60}", end="\r")
                print(f"Seq2Seq: Epoch {epoch + 1}/{epochs} Loss {total_loss:.6f} Avg {avg:.6f}")

        return history

    @staticmethod
    def _print_progress(current, total, bar_len=30):
        filled = int(bar_len * current / max(total, 1))
        bar = "█" * filled + "-" * (bar_len - filled)
        percent = (current / max(total, 1)) * 100
        print(f"\r[{bar}] {percent:5.1f}%", end="")

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
