# Machine Translation for English based on Long Short Term Memory
This repository ...

## 0. Require
```zsh
brew install python@3.11
```

## 1. Preparation
### 1.1 Create and active a venv
```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1.2 Install
```sh
pip install -r requirements.txt
```

## 1.3 Datasets
You can update dataset for English/Vietnam language on `/datasets/src_en.txt` and `/datasets/tgt_vi.txt`

## 2. Train

## 3. For developer
### 3.1 Check code
```py
python scripts/check_code.py
```

### 3.2 Clean code
```py
python scripts/clean.py
```

### 3.3 Format code
```py
python scripts/format_code.py
```

## Acknowledgement
