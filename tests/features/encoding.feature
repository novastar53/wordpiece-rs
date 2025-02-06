Feature: Token Encoding and Decoding
  As a NLP developer
  I want to convert between text and token IDs
  So that I can interface with neural networks

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

  Scenario: Text to token IDs
    When I encode the text "wanted to go home"
    Then I should get the following token IDs
      | id |
      | 3  |
      | 4  |
      | 5  |
      | 6  |
      | 7  |

  Scenario: Token IDs to text
    When I decode the token IDs [3, 4, 5, 6, 7]
    Then I should get the text "wanted to go home"

  Scenario: Round trip conversion
    When I encode the text "wanted to go home"
    And I decode the resulting token IDs
    Then I should get the original text back

  Scenario: Handling unknown tokens in encoding
    When I encode the text "unknown word"
    Then I should get the following token IDs
      | id |
      | 0  |
      | 0  |

  Scenario: Special token preservation in decoding
    When I decode the token IDs [1, 3, 4, 5, 6, 7, 2]
    Then I should get the text "[CLS] wanted to go home [SEP]"