def read_file(file_path) -> str:
    with open(file_path, "r") as file:
        return file.read()


def to_lines(text) -> list:
    sentences = text.strip().split("\n")
    sentences = [i.split("\t") for i in sentences]
    return sentences
