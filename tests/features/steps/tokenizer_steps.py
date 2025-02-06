from behave import given, when, then
import wordpiece_rs
from typing import Dict, List
import json

@given('a basic vocabulary with the following tokens')
def step_impl(context):
    vocab = {}
    for row in context.table:
        vocab[row['token']] = int(row['id'])
    context.tokenizer = wordpiece_rs.WordPieceTokenizer(vocab)

@given('a tokenizer with case_sensitive={case_sensitive}')
def step_impl(context, case_sensitive):
    if not hasattr(context, 'tokenizer'):
        vocab = {
            "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
            "want": 3, "##ed": 4, "to": 5, "go": 6, "home": 7
        }
        context.tokenizer = wordpiece_rs.WordPieceTokenizer(
            vocab,
            lowercase=not json.loads(case_sensitive.lower())
        )

@when('I tokenize the text "{text}"')
def step_impl(context, text):
    context.tokens = context.tokenizer.tokenize(text)

@then('I should get the following tokens')
def step_impl(context):
    expected_tokens = [row['token'] for row in context.table]
    assert context.tokens == expected_tokens, \
        f"Expected {expected_tokens}, but got {context.tokens}"

@when('I encode the text "{text}"')
def step_impl(context, text):
    context.token_ids = context.tokenizer.encode(text)

@when('I decode the token IDs {ids}')
def step_impl(context, ids):
    token_ids = eval(ids)  # Safe in test context
    context.decoded_text = context.tokenizer.decode(token_ids)

@when('I decode the resulting token IDs')
def step_impl(context):
    context.decoded_text = context.tokenizer.decode(context.token_ids)

@then('I should get the following token IDs')
def step_impl(context):
    expected_ids = [int(row['id']) for row in context.table]
    assert context.token_ids == expected_ids, \
        f"Expected {expected_ids}, but got {context.token_ids}"

@then('I should get the text "{text}"')
def step_impl(context, text):
    assert context.decoded_text == text, \
        f"Expected '{text}', but got '{context.decoded_text}'"

@then('I should get the original text back')
def step_impl(context):
    assert hasattr(context, 'decoded_text'), "No decoded text available"
    original_text = context.text.lower() if not context.tokenizer.lowercase else context.text
    assert context.decoded_text == original_text, \
        f"Expected '{original_text}', but got '{context.decoded_text}'"