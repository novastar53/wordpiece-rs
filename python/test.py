import wordpiece_rs

# Example vocabulary
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

# Create tokenizer
tokenizer = wordpiece_rs.WordPieceTokenizer(vocab)

# Test tokenization
text = "wanted to go home"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Test encoding
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")

# Test decoding
decoded = tokenizer.decode(ids)
print(f"Decoded text: {decoded}")