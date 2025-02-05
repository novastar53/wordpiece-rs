use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use regex::{Regex, RegexBuilder};
use std::borrow::Cow;

#[derive(Default)]
struct TrieNode {
    """
    Define a TrieNode struct to represent a node in the Trie data structure.
    """
    children: HashMap<char, TrieNode>, // A map of child nodes, where each child node is represented by a character
    is_word: bool, // Indicates whether the node represents the end of a word
    token_id: i32, // The token ID associated with the word
}

impl TrieNode {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, word: &str, token_id: i32) {
        """ 
        Insert a word into the Trie.
        """
        let mut node = self;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        } // Traverse the Trie by iterating over each character in the word
        node.is_word = true; // Mark the last node as the end of a word
        node.token_id = token_id; // Store the token ID associated with the word
    }

    fn find_longest_prefix(&self, word: &[char], start: usize) -> Option<(usize, i32)> {
        """
        Find the longest prefix of a word in the Trie.
        """
        let mut node = self;
        let mut last_match = None;
        let mut pos = start;

        while pos < word.len() {
            if let Some(next) = node.children.get(&word[pos]) {
                if next.is_word {
                    last_match = Some((pos + 1, next.token_id));
                }
                node = next;
                pos += 1;
            } else {
                break;
            }
        }

        last_match
    }
}

/// Token represents a single token with its text, ID, and whether it's a special token
#[derive(Debug, Clone)]
struct Token {
    text: String,
    id: i32,
    is_special: bool,
}

#[pyclass]
struct WordPieceTokenizer {
    trie: TrieNode,
    vocab_lookup: HashMap<i32, String>,
    unk_token: String,
    unk_token_id: i32,
    max_input_chars_per_word: usize,
    special_tokens: HashMap<String, i32>,
    basic_tokenizer: Regex,
    punctuation: Regex,
    chinese_chars: Regex,
    strip_accents: bool,
    lowercase: bool,
}

#[pymethods]
impl WordPieceTokenizer {
    #[new]
    #[args(
        unk_token = "\"[UNK]\"",
        max_input_chars_per_word = "200",
        strip_accents = "true",
        lowercase = "true"
    )]
    fn new(
        vocab: &PyDict,
        unk_token: &str,
        max_input_chars_per_word: usize,
        strip_accents: bool,
        lowercase: bool,
    ) -> Self {
        let mut trie = TrieNode::new();
        let mut vocab_lookup = HashMap::new();
        let mut special_tokens = HashMap::new();
        let unk = unk_token.to_string();
        let mut unk_id = 0;

        // Compile regex patterns
        let basic_tokenizer = RegexBuilder::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .case_insensitive(true)
            .build()
            .unwrap();
        
        let punctuation = RegexBuilder::new(r"\p{P}")
            .build()
            .unwrap();

        let chinese_chars = RegexBuilder::new(r"[\p{Script=Han}]")
            .build()
            .unwrap();

        // Process vocabulary
        for (k, v) in vocab.iter() {
            let key = k.extract::<String>().unwrap();
            let value = v.extract::<i32>().unwrap();
            
            if key == unk {
                unk_id = value;
            }
            
            // Identify special tokens (those that don't start with ## and contain special chars)
            if !key.starts_with("##") && (key.starts_with('[') || key.starts_with('<') || punctuation.is_match(&key)) {
                special_tokens.insert(key.clone(), value);
            } else {
                trie.insert(&key, value);
            }
            
            vocab_lookup.insert(value, key);
        }

        WordPieceTokenizer {
            trie,
            vocab_lookup,
            unk_token: unk,
            unk_token_id: unk_id,
            max_input_chars_per_word,
            special_tokens,
            basic_tokenizer,
            punctuation,
            chinese_chars,
            strip_accents,
            lowercase,
        }
    }

    fn clean_text(&self, text: &str) -> String {
        // Normalize unicode characters
        let text = text.nfkc().collect::<String>();
        
        // Replace whitespace characters with space
        let text = text.replace(|c: char| c.is_whitespace(), " ");
        
        // Handle Chinese characters by adding spaces around them
        let text = self.chinese_chars.replace_all(&text, |caps: &regex::Captures| {
            format!(" {} ", &caps[0])
        }).into_owned();
        
        text
    }

    fn strip_accents_if_needed(&self, text: &str) -> Cow<str> {
        if !self.strip_accents {
            return Cow::Borrowed(text);
        }

        let normalized = text.nfd().collect::<String>();
        let stripped = normalized
            .chars()
            .filter(|&c| !char::is_mark(c))
            .collect::<String>();
        Cow::Owned(stripped)
    }

    fn basic_tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let text = self.clean_text(text);
        
        for mat in self.basic_tokenizer.find_iter(&text) {
            let mut token_text = mat.as_str().trim().to_string();
            
            // Check if it's a special token
            if let Some(&id) = self.special_tokens.get(&token_text) {
                tokens.push(Token {
                    text: token_text,
                    id,
                    is_special: true,
                });
                continue;
            }
            
            // Handle casing
            if self.lowercase {
                token_text = token_text.to_lowercase();
            }
            
            // Handle accents
            token_text = self.strip_accents_if_needed(&token_text).into_owned();
            
            // Split on punctuation
            let mut char_tokens = Vec::new();
            let mut current = String::new();
            
            for c in token_text.chars() {
                if self.punctuation.is_match(&c.to_string()) {
                    if !current.is_empty() {
                        char_tokens.push(current);
                        current = String::new();
                    }
                    char_tokens.push(c.to_string());
                } else {
                    current.push(c);
                }
            }
            
            if !current.is_empty() {
                char_tokens.push(current);
            }
            
            // Create tokens
            for t in char_tokens {
                tokens.push(Token {
                    text: t,
                    id: -1, // Will be assigned during wordpiece tokenization
                    is_special: false,
                });
            }
        }
        
        tokens
    }

    fn wordpiece_tokenize(&self, token: &Token) -> Vec<Token> {
        if token.is_special {
            return vec![token.clone()];
        }

        let chars: Vec<char> = token.text.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return vec![Token {
                text: self.unk_token.clone(),
                id: self.unk_token_id,
                is_special: true,
            }];
        }

        let mut start = 0;
        let mut sub_tokens = Vec::new();
        let mut is_bad = false;

        while start < chars.len() {
            let prefix = if start == 0 {
                self.trie.find_longest_prefix(&chars, 0)
            } else {
                let mut prefix_chars = Vec::with_capacity(2 + chars.len() - start);
                prefix_chars.extend(['#', '#']);
                prefix_chars.extend(&chars[start..]);
                self.trie.find_longest_prefix(&prefix_chars, 0)
            };

            if let Some((len, token_id)) = prefix {
                let token_text = self.vocab_lookup.get(&token_id).unwrap().clone();
                sub_tokens.push(Token {
                    text: token_text,
                    id: token_id,
                    is_special: false,
                });
                start += if start == 0 { len } else { len - 2 };
            } else {
                is_bad = true;
                break;
            }
        }

        if is_bad {
            vec![Token {
                text: self.unk_token.clone(),
                id: self.unk_token_id,
                is_special: true,
            }]
        } else {
            sub_tokens
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        // First apply basic tokenization
        let basic_tokens = self.basic_tokenize(text);
        
        // Then apply WordPiece tokenization to each token
        basic_tokens
            .into_iter()
            .flat_map(|token| self.wordpiece_tokenize(&token))
            .map(|token| token.text)
            .collect()
    }

    fn encode(&self, text: &str) -> Vec<i32> {
        // First apply basic tokenization
        let basic_tokens = self.basic_tokenize(text);
        
        // Then apply WordPiece tokenization to each token
        basic_tokens
            .into_iter()
            .flat_map(|token| self.wordpiece_tokenize(&token))
            .map(|token| token.id)
            .collect()
    }

    fn decode(&self, ids: Vec<i32>) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.vocab_lookup.get(&id))
            .map(|t| t.replace("##", ""))
            .collect();

        // Join tokens with spaces, but don't add spaces around punctuation
        let mut result = String::new();
        let mut prev_is_punct = false;
        
        for (i, token) in tokens.iter().enumerate() {
            let is_punct = self.punctuation.is_match(token);
            
            if i > 0 && !is_punct && !prev_is_punct {
                result.push(' ');
            }
            
            result.push_str(token);
            prev_is_punct = is_punct;
        }
        
        result
    }
}

#[pymodule]
fn wordpiece_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WordPieceTokenizer>()?;
    Ok(())
}