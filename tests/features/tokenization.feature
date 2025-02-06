Feature: WordPiece Tokenization
  As a NLP developer
  I want to tokenize text using WordPiece algorithm
  So that I can prepare text for language model input

  Background:
    Given a basic vocabulary with the following tokens
      | token  | id |
      | [UNK]  | 0  |
      | [CLS]  | 1  |
      | [SEP]  | 2  |
      | want   | 3  |
      | ##ed   | 4  |
      | to     | 5  |
      | go     | 6  |
      | home   | 7  |
      | play   | 8  |
      | ##ing  | 9  |
      | !      | 10 |
      | 我     | 11 |
      | 想     | 12 |
      | 回     | 13 |
      | 家     | 14 |

  Scenario: Basic tokenization
    When I tokenize the text "wanted to go home"
    Then I should get the following tokens
      | token |
      | want  |
      | ##ed  |
      | to    |
      | go    |
      | home  |

  Scenario: Handling unknown tokens
    When I tokenize the text "unknown word"
    Then I should get the following tokens
      | token  |
      | [UNK]  |
      | [UNK]  |

  Scenario: Mixed language tokenization
    When I tokenize the text "I want to go 回家!"
    Then I should get the following tokens
      | token  |
      | [UNK]  |
      | want   |
      | to     |
      | go     |
      | 回     |
      | 家     |
      | !      |

  Scenario: Case sensitivity
    Given a tokenizer with case_sensitive=false
    When I tokenize the text "WANTED to GO home"
    Then I should get the following tokens
      | token |
      | want  |
      | ##ed  |
      | to    |
      | go    |
      | home  |

  Scenario: Special token handling
    When I tokenize the text "[CLS] wanted to go home [SEP]"
    Then I should get the following tokens
      | token |
      | [CLS] |
      | want  |
      | ##ed  |
      | to    |
      | go    |
      | home  |
      | [SEP] |