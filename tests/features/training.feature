Feature: WordPiece Vocabulary Training
  As a NLP developer
  I want to train a WordPiece vocabulary from text
  So that I can create custom tokenizers for my domain

  Background:
    Given a training corpus
      | text                                |
      | the quick brown fox jumps          |
      | pack my box with five dozen jugs   |
      | how vexingly quick daft zebras     |
      | the five boxing wizards jump       |
      | sphinx of black quartz judge       |
      | waltz nymph for quick jigs         |
      | quick zephyrs blow vexing          |
      | two driven jocks help fax          |
      | the jay pig fox zebra wolves       |
      | watch jeopardy alex fun quiz       |

  Scenario: Basic vocabulary training
    When I train a vocabulary with size=100 and min_frequency=2
    Then the vocabulary should contain special tokens
      | token  |
      | [UNK]  |
      | [CLS]  |
      | [SEP]  |
      | [PAD]  |
      | [MASK] |
    And the vocabulary should contain common subwords
      | token |
      | quick |
      | the   |
      | fox   |
      | ##ing |
    And the vocabulary size should be 100

  Scenario: Training with custom parameters
    When I train a vocabulary with the following parameters
      | parameter      | value |
      | vocab_size    | 50    |
      | min_frequency | 3     |
    Then the vocabulary size should be 50
    And all tokens should appear at least 3 times in the corpus

  Scenario: Training with custom special tokens
    Given a list of custom special tokens
      | token   |
      | <start> |
      | <end>   |
      | <pad>   |
    When I train a vocabulary with size=100
    Then the vocabulary should contain these special tokens
    And these tokens should have the first IDs