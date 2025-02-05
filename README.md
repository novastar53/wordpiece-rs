# WordPiece Tokenizer in Rust

A fast implementation of the WordPiece tokenizer in Rust with Python bindings using PyO3.

## Key Features

- Fast tokenization using Rust
- Python bindings via PyO3
- Unicode normalization (NFKC)
- Configurable unknown token and maximum input length
- Support for custom vocabularies

## Installation

### From Source

1. Make sure you have Rust and Python installed
2. Install maturin: `pip install maturin`
3. Build and install:
```bash
cd wordpiece_rs
maturin develop
```

## Usage

```python
import wordpiece_rs

# Create a vocabulary (token -> id mapping)
vocab = {
    "[UNK]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "want": 3,
    "##ed": 4,
    "to": 5,
    "go": 6,
    "home": 7,
}

# Initialize the tokenizer
tokenizer = wordpiece_rs.WordPieceTokenizer(vocab)

# Tokenize text
tokens = tokenizer.tokenize("wanted to go home")
print(tokens)  # ['want', '##ed', 'to', 'go', 'home']

# Encode text to token IDs
ids = tokenizer.encode("wanted to go home")
print(ids)  # [3, 4, 5, 6, 7]

# Decode token IDs back to text
text = tokenizer.decode([3, 4, 5, 6, 7])
print(text)  # "wantedtogohome"
```

## Customization

You can customize the tokenizer by providing optional parameters:

```python
tokenizer = wordpiece_rs.WordPieceTokenizer(
    vocab,
    unk_token="<UNK>",  # Default: "[UNK]"
    max_input_chars_per_word=100  # Default: 200
)
```

## License

MIT License