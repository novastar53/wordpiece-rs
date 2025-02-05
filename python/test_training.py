import wordpiece_rs

# Example training data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "The five boxing wizards jump quickly",
    "Sphinx of black quartz, judge my vow",
    "Waltz, nymph, for quick jigs vex Bud",
    "Quick zephyrs blow, vexing daft Jim",
    "Two driven jocks help fax my big quiz",
    "The jay, pig, fox, zebra and my wolves quack",
    "Watch Jeopardy!, Alex Trebek's fun TV quiz game",
]

# Train vocabulary
vocab = wordpiece_rs.WordPieceTokenizer.train(
    texts=texts,
    vocab_size=100,  # Small size for demonstration
    min_frequency=2,  # Minimum frequency for a token
    strip_accents=True,
    lowercase=True,
)

print("Trained vocabulary:")
for token, id in sorted(vocab.items(), key=lambda x: x[1]):
    print(f"{id:3d}: {token}")

# Create tokenizer with trained vocabulary
tokenizer = wordpiece_rs.WordPieceTokenizer(vocab)

# Test tokenization
test_text = "The quick brown fox jumps!"
tokens = tokenizer.tokenize(test_text)
print(f"\nTokenizing: {test_text}")
print(f"Tokens: {tokens}")

# Test encoding
ids = tokenizer.encode(test_text)
print(f"Token IDs: {ids}")

# Test decoding
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")