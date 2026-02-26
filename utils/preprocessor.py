import string

import nltk
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from underthesea import text_normalize as underthesea_text_normalize
from underthesea import word_tokenize as underthesea_word_tokenize

from constants.common import EOS_TOKEN, SOS_TOKEN

nltk.download("punkt", quiet=True)


class Preprocessor:
    def __init__(self, language="english"):
        self.language = language

    def preprocess(self, text):
        if self.language == "english":
            return self._preprocess_english(text)
        elif self.language == "vietnamese":
            return self._preprocess_vietnamese(text)
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _preprocess_english(self, text) -> list:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = nltk_word_tokenize(text)
        return tokens

    def _preprocess_vietnamese(self, text) -> list:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = underthesea_text_normalize(text)
        tokens = underthesea_word_tokenize(text, format="text")
        tokens = [SOS_TOKEN] + tokens.split() + [EOS_TOKEN]
        return tokens
