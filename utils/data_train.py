def split_data(data, train_size=0.8):
    train_data = data.sample(frac=train_size, random_state=42)
    test_data = data.drop(train_data.index)
    return train_data, test_data


def encode_sentence(x, word2idx):
    if isinstance(x, str):
        tokens = x.split()
    else:
        tokens = x

    return [word2idx[w] for w in tokens if w in word2idx]


def decode_sequence(indices, idx_to_word):
    words = []
    for idx in indices:
        if idx in idx_to_word:
            words.append(idx_to_word[idx])
    return " ".join(words)
