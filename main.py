import math
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd

from layers.decoder import Decoder
from layers.encoder import Encoder
from layers.lstm_cell import LstmCell
from models.seq2seq import Seq2Seq
from models.skip_gram import SkipGram
from utils.data_train import decode_sequence, encode_sentence, split_data
from utils.file import read_file, to_lines
from utils.preprocessor import Preprocessor


class Main:
    def __init__(self):
        pass

    def run(self):
        config = {
            "embedding_dim": 100,
            "hidden_size": 128,
            "skipgram_epochs": 10,
            "skipgram_batch_size": 256,
            "seq2seq_epochs": 30,
            "seq2seq_batch_size": 32,
            "lr": 0.01,
            "bleu_samples": 200,
        }

        vi_text, en_text = self._load_data()
        parsed = self._run_parallel(
            {
                "vi": lambda: to_lines(vi_text),
                "en": lambda: to_lines(en_text),
            }
        )
        arr_vi_text = parsed["vi"]
        arr_en_text = parsed["en"]

        self._plt_token_totals(arr_en_text, arr_vi_text)

        vi_train_data, vi_test_data = self._train_test_split(arr_vi_text)
        en_train_data, en_test_data = self._train_test_split(arr_en_text)

        tokenized = self._run_parallel(
            {
                "vi_train": lambda: self._tokenize(vi_train_data, language="vietnamese"),
                "en_train": lambda: self._tokenize(en_train_data, language="english"),
                "vi_test": lambda: self._tokenize(vi_test_data, language="vietnamese"),
                "en_test": lambda: self._tokenize(en_test_data, language="english"),
            }
        )
        vi_tokens_train = tokenized["vi_train"]
        en_tokens_train = tokenized["en_train"]
        vi_tokens_test = tokenized["vi_test"]
        en_tokens_test = tokenized["en_test"]

        print(f"Sample tokenized Vietnamese train: {vi_tokens_train[0]}")
        print(f"Sample tokenized English train: {en_tokens_train[0]}")
        skipgrams = self._run_parallel(
            {
                "vi": lambda: self._train_skipgram(
                    vi_tokens_train,
                    embedding_dim=config["embedding_dim"],
                    epochs=config["skipgram_epochs"],
                    batch_size=config["skipgram_batch_size"],
                ),
                "en": lambda: self._train_skipgram(
                    en_tokens_train,
                    embedding_dim=config["embedding_dim"],
                    epochs=config["skipgram_epochs"],
                    batch_size=config["skipgram_batch_size"],
                ),
            }
        )
        vi_skip_gram, vi_embeddings = skipgrams["vi"]
        en_skip_gram, en_embeddings = skipgrams["en"]

        SOS_IDX = vi_skip_gram.word_to_idx.get("<SOS>")
        EOS_IDX = vi_skip_gram.word_to_idx.get("<EOS>")
        if SOS_IDX is None or EOS_IDX is None:
            raise ValueError(
                "TARGET vocab missing <SOS>/<EOS>. Ensure vi_tokens_train has them before SkipGram.train()."
            )

        models = self._run_parallel(
            {
                "encoder": lambda: self._create_encoder(en_embeddings, config),
                "decoder": lambda: self._create_decoder(vi_embeddings, len(vi_skip_gram.word_to_idx), config),
            }
        )
        encoder = models["encoder"]
        decoder = models["decoder"]

        model = Seq2Seq(encoder, decoder, lr=config["lr"], sos_idx=SOS_IDX, eos_idx=EOS_IDX)

        dataset_train = self._get_dataset(en_tokens_train, vi_tokens_train, en_skip_gram, vi_skip_gram)
        dataset_test = self._get_dataset(en_tokens_test, vi_tokens_test, en_skip_gram, vi_skip_gram)

        history = model.train(
            dataset_train,
            epochs=config["seq2seq_epochs"],
            batch_size=config["seq2seq_batch_size"],
            val_dataset=dataset_test,
        )
        self._plt_loss_curve(history)
        self._evaluate_samples(model, dataset_test, vi_skip_gram.idx_to_word)
        self._token_accuracy(model, dataset_test)
        self._plt_bleu_scores(model, dataset_test, vi_skip_gram.idx_to_word, n=config["bleu_samples"])

        self._df_data(arr_vi_text, arr_en_text)
        self._df_tokenize(vi_tokens_train, en_tokens_train)

    def _token_accuracy(self, model, dataset):
        correct = 0
        total = 0

        for src_indices, trg_full in dataset:
            pred = model.translate(src_indices)

            trg = trg_full[1:-1]

            for p, t in zip(pred, trg):
                if p == t:
                    correct += 1
                total += 1

        print("Token accuracy:", correct / max(total, 1))

    def _evaluate_samples(self, model, dataset, idx_to_word, n=5):
        print("\n=== Evaluation Samples ===")

        for src_indices, trg_full in dataset[:n]:

            pred = model.translate(src_indices, max_len=30)

            trg_clean = [t for t in trg_full if idx_to_word.get(t) not in ("<SOS>", "<EOS>")]

            print("PRED:", decode_sequence(pred, idx_to_word))
            print("TRUE:", decode_sequence(trg_clean, idx_to_word))
            print("-" * 40)

    def _get_dataset(self, en_tokens, vi_tokens, en_sg, vi_sg):
        dataset = []
        for en_s, vi_s in zip(en_tokens, vi_tokens):
            src_indices = encode_sentence(en_s, en_sg.word_to_idx)
            trg_indices = encode_sentence(vi_s, vi_sg.word_to_idx)
            dataset.append((src_indices, trg_indices))
        return dataset

    def _train_test_split(self, arr_text, train_size=0.8) -> tuple[list[str], list[str]]:
        data = pd.DataFrame(arr_text, columns=["text"])
        train_data, test_data = split_data(data, train_size=train_size)

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        return train_data["text"].tolist(), test_data["text"].tolist()

    def _tokenize(self, arr_text, language="english") -> list[str]:
        preprocessor = Preprocessor(language=language)

        tokens = []
        for text in arr_text:
            token = preprocessor.preprocess(text)
            tokens.append(" ".join(token))
        return tokens

    def _df_tokenize(self, vi_tokens, en_tokens) -> None:
        vi_df = pd.DataFrame(vi_tokens, columns=["Vietnamese"])
        eng_df = pd.DataFrame(en_tokens, columns=["English"])
        vi_eng = pd.concat([eng_df, vi_df], axis=1)
        print(vi_eng.head())

    def _load_data(self) -> tuple[str, str]:
        vi_text = read_file("datasets/tgt_vi.txt")
        en_text = read_file("datasets/src_en.txt")
        return vi_text, en_text

    def _run_parallel(self, tasks: dict[str, callable]) -> dict[str, object]:
        if not tasks:
            return {}
        max_workers = min(len(tasks), 8)
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in futures:
                name = futures[future]
                results[name] = future.result()
        return results

    def _train_skipgram(self, tokens, embedding_dim, epochs, batch_size):
        skip_gram = SkipGram(embedding_dim=embedding_dim)
        skip_gram.train(tokens, epochs=epochs, batch_size=batch_size)
        embeddings = skip_gram.get_embedding_matrix()
        return skip_gram, embeddings

    def _create_encoder(self, en_embeddings, config):
        encoder_cell = LstmCell(input_size=config["embedding_dim"], hidden_size=config["hidden_size"])
        return Encoder(encoder_cell, en_embeddings)

    def _create_decoder(self, vi_embeddings, vocab_size, config):
        decoder_cell = LstmCell(
            input_size=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            output_size=vocab_size,
        )
        return Decoder(decoder_cell, vi_embeddings)

    def _df_data(self, arr_vi_text, arr_en_text) -> None:
        vi_df = pd.DataFrame(arr_vi_text, columns=["Vietnamese"])
        eng_df = pd.DataFrame(arr_en_text, columns=["English"])

        print(f"Vietnamese shape: {vi_df.shape}")
        print(f"English shape: {eng_df.shape}")

        vi_eng = pd.concat([eng_df, vi_df], axis=1)
        print(vi_eng.head())

    def _plt_token_totals(self, eng_texts, vi_texts) -> None:
        token_counts = [
            sum(len(text.split()) if isinstance(text, str) else len(text) for text in eng_texts),
            sum(len(text.split()) if isinstance(text, str) else len(text) for text in vi_texts),
        ]

        plt.figure(figsize=(6, 4))
        plt.bar(["English", "Vietnamese"], token_counts, color=["#4C78A8", "#F58518"])
        plt.ylabel("Total Tokens")
        plt.title("Total Tokens by Language")
        self._save_fig("token_totals_bar.png")
        plt.show()

    def _save_fig(self, filename) -> None:
        os.makedirs("figures", exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", filename), dpi=150)

    def _plt_loss_curve(self, history: dict) -> None:
        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])

        if not train_losses:
            return

        epochs = list(range(1, len(train_losses) + 1))

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label="Train", color="#4C78A8")

        if val_losses:
            plt.plot(epochs, val_losses, label="Val", color="#F58518")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Seq2Seq Loss")
        plt.legend()
        self._save_fig("seq2seq_loss.png")
        plt.show()

    def _plt_bleu_scores(self, model, dataset, idx_to_word, n=200) -> None:
        if not dataset:
            return

        samples = dataset[: min(n, len(dataset))]
        bleu_scores = []

        for src_indices, trg_full in samples:
            pred = model.translate(src_indices, max_len=30)
            pred_tokens = decode_sequence(pred, idx_to_word).split()
            trg_tokens = decode_sequence(
                [t for t in trg_full if idx_to_word.get(t) not in ("<SOS>", "<EOS>")],
                idx_to_word,
            ).split()

            bleu_scores.append(self._sentence_bleu(pred_tokens, trg_tokens))

        plt.figure(figsize=(6, 4))
        plt.hist(bleu_scores, bins=20, color="#4C78A8", alpha=0.85)
        plt.xlabel("BLEU")
        plt.ylabel("Count")
        plt.title("Seq2Seq BLEU Score Distribution")
        self._save_fig("seq2seq_bleu.png")
        plt.show()

    def _sentence_bleu(self, pred_tokens, ref_tokens, max_n=4, smooth=1.0) -> float:
        if not pred_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = self._ngram_counts(pred_tokens, n)
            ref_ngrams = self._ngram_counts(ref_tokens, n)

            match = 0
            total = 0
            for ngram, count in pred_ngrams.items():
                total += count
                match += min(count, ref_ngrams.get(ngram, 0))

            precisions.append((match + smooth) / (total + smooth))

        log_precision = sum(math.log(p) for p in precisions) / max_n
        bp = 1.0
        ref_len = len(ref_tokens)
        pred_len = len(pred_tokens)
        if pred_len < ref_len:
            bp = math.exp(1 - (ref_len / max(pred_len, 1)))

        return bp * math.exp(log_precision)

    def _ngram_counts(self, tokens, n):
        counts = {}
        if n <= 0 or len(tokens) < n:
            return counts
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    def _plt_tokenize(self, eng_tokens, vi_tokens) -> None:
        eng_l = [len(text.split()) for text in eng_tokens]
        vi_l = [len(text.split()) for text in vi_tokens]

        length_df = pd.DataFrame({"eng": eng_l, "vi": vi_l})
        length_df.hist(bins=30)
        plt.show()


if __name__ == "__main__":
    main = Main()
    main.run()
