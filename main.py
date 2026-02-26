import os

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
        vi_text, en_text = self._load_data()
        arr_vi_text = to_lines(vi_text)
        arr_en_text = to_lines(en_text)

        self._plt_token_totals(arr_en_text, arr_vi_text)

        vi_train_data, vi_test_data = self._train_test_split(arr_vi_text)
        en_train_data, en_test_data = self._train_test_split(arr_en_text)

        vi_tokens_train = self._tokenize(vi_train_data, language="vietnamese")
        en_tokens_train = self._tokenize(en_train_data, language="english")
        vi_tokens_test = self._tokenize(vi_test_data, language="vietnamese")
        en_tokens_test = self._tokenize(en_test_data, language="english")

        vi_skip_gram = SkipGram(embedding_dim=100)
        vi_skip_gram.train(vi_tokens_train)
        vi_embeddings = vi_skip_gram.get_embedding_matrix()

        en_skip_gram = SkipGram(embedding_dim=100)
        en_skip_gram.train(en_tokens_train)
        en_embeddings = en_skip_gram.get_embedding_matrix()

        SOS_IDX = vi_skip_gram.word_to_idx.get("<SOS>")
        EOS_IDX = vi_skip_gram.word_to_idx.get("<EOS>")
        if SOS_IDX is None or EOS_IDX is None:
            raise ValueError(
                "TARGET vocab missing <SOS>/<EOS>. Ensure vi_tokens_train has them before SkipGram.train()."
            )

        encoder_cell = LstmCell(input_size=100, hidden_size=128)
        decoder_cell = LstmCell(input_size=100, hidden_size=128, output_size=len(vi_skip_gram.word_to_idx))

        encoder = Encoder(encoder_cell, en_embeddings)
        decoder = Decoder(decoder_cell, vi_embeddings)

        model = Seq2Seq(encoder, decoder, lr=0.01, sos_idx=SOS_IDX, eos_idx=EOS_IDX)

        dataset_train = self._get_dataset(en_tokens_train, vi_tokens_train, en_skip_gram, vi_skip_gram)
        dataset_test = self._get_dataset(en_tokens_test, vi_tokens_test, en_skip_gram, vi_skip_gram)

        model.train(dataset_train, epochs=30)
        self._evaluate_samples(model, dataset_test, vi_skip_gram.idx_to_word)
        self._token_accuracy(model, dataset_test)

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

    def _plt_tokenize(self, eng_tokens, vi_tokens) -> None:
        eng_l = [len(text.split()) for text in eng_tokens]
        vi_l = [len(text.split()) for text in vi_tokens]

        length_df = pd.DataFrame({"eng": eng_l, "vi": vi_l})
        length_df.hist(bins=30)
        plt.show()


if __name__ == "__main__":
    main = Main()
    main.run()
