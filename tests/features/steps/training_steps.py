from behave import given, when, then
import wordpiece_rs
from typing import Dict, List, Set

@given('a training corpus')
def step_impl(context):
    context.corpus = [row['text'] for row in context.table]

@given('a list of custom special tokens')
def step_impl(context):
    context.special_tokens = [row['token'] for row in context.table]

@when('I train a vocabulary with size={size:d} and min_frequency={freq:d}')
def step_impl(context, size: int, freq: int):
    context.vocab = wordpiece_rs.WordPieceTokenizer.train(
        texts=context.corpus,
        vocab_size=size,
        min_frequency=freq
    )

@when('I train a vocabulary with the following parameters')
def step_impl(context):
    params = {row['parameter']: int(row['value']) for row in context.table}
    context.vocab = wordpiece_rs.WordPieceTokenizer.train(
        texts=context.corpus,
        vocab_size=params['vocab_size'],
        min_frequency=params['min_frequency']
    )

@then('the vocabulary should contain special tokens')
def step_impl(context):
    special_tokens = {row['token'] for row in context.table}
    vocab_tokens = set(context.vocab.keys())
    missing_tokens = special_tokens - vocab_tokens
    assert not missing_tokens, f"Missing special tokens: {missing_tokens}"

@then('the vocabulary should contain common subwords')
def step_impl(context):
    expected_tokens = {row['token'] for row in context.table}
    vocab_tokens = set(context.vocab.keys())
    missing_tokens = expected_tokens - vocab_tokens
    assert not missing_tokens, f"Missing common subwords: {missing_tokens}"

@then('the vocabulary size should be {size:d}')
def step_impl(context, size: int):
    assert len(context.vocab) == size, \
        f"Expected vocabulary size {size}, but got {len(context.vocab)}"

@then('all tokens should appear at least {freq:d} times in the corpus')
def step_impl(context, freq: int):
    # Create a tokenizer with the trained vocabulary
    tokenizer = wordpiece_rs.WordPieceTokenizer(context.vocab)
    
    # Count token occurrences
    token_counts: Dict[str, int] = {}
    for text in context.corpus:
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            if not token.startswith('['):  # Skip special tokens
                token_counts[token] = token_counts.get(token, 0) + 1
    
    # Check frequencies
    low_freq_tokens = {
        token: count for token, count in token_counts.items()
        if count < freq and not token.startswith('[')
    }
    assert not low_freq_tokens, \
        f"Found tokens with frequency < {freq}: {low_freq_tokens}"

@then('the vocabulary should contain these special tokens')
def step_impl(context):
    vocab_tokens = set(context.vocab.keys())
    for token in context.special_tokens:
        assert token in vocab_tokens, f"Missing special token: {token}"

@then('these tokens should have the first IDs')
def step_impl(context):
    special_token_ids = sorted(
        context.vocab[token] for token in context.special_tokens
    )
    expected_ids = list(range(len(context.special_tokens)))
    assert special_token_ids == expected_ids, \
        f"Special tokens don't have first IDs. Expected {expected_ids}, got {special_token_ids}"