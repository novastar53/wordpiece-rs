use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use regex::Regex;

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

#[pyclass]
struct WordPieceTokenizer {
    trie: TrieNode,
    vocab_lookup: HashMap<i32, String>,
    unk_token: String,
    unk_token_id: i32,
    max_input_chars_per_word: usize,
}

#[pymethods]
impl WordPieceTokenizer {
    #[new]
    fn new(vocab: &PyDict, unk_token: Option<&str>, max_input_chars_per_word: Option<usize>) -> Self {
        let mut trie = TrieNode::new();
        let mut vocab_lookup = HashMap::new();
        let unk = unk_token.unwrap_or("[UNK]").to_string();
        let mut unk_id = 0;

        for (k, v) in vocab.iter() {
            let key = k.extract::<String>().unwrap();
            let value = v.extract::<i32>().unwrap();
            if key == unk {
                unk_id = value;
            }
            trie.insert(&key, value);
            vocab_lookup.insert(value, key);
        }

        WordPieceTokenizer {
            trie,
            vocab_lookup,
            unk_token: unk,
            unk_token_id: unk_id,
            max_input_chars_per_word: max_input_chars_per_word.unwrap_or(200),
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let re = Regex::new(r"[^\s]+").unwrap();
        let text = text.nfkc().collect::<String>();
        let mut output_tokens = Vec::new();

        for token in re.find_iter(&text) {
            let chars: Vec<char> = token.as_str().chars().collect();
            if chars.len() > self.max_input_chars_per_word {
                output_tokens.push(self.unk_token.clone());
                continue;
            }

            let mut start = 0;
            let mut sub_tokens = Vec::new();
            let mut is_bad = false;

            while start < chars.len() {
                let prefix = if start == 0 {
                    // For the first token, look in the trie directly
                    self.trie.find_longest_prefix(&chars, 0)
                } else {
                    // For subsequent tokens, prepend "##" and look in the trie
                    let mut prefix_chars = Vec::with_capacity(2 + chars.len() - start);
                    prefix_chars.extend(['#', '#']);
                    prefix_chars.extend(&chars[start..]);
                    self.trie.find_longest_prefix(&prefix_chars, 0)
                };

                if let Some((len, token_id)) = prefix {
                    let token = self.vocab_lookup.get(&token_id).unwrap().clone();
                    sub_tokens.push(token);
                    start += if start == 0 { len } else { len - 2 };
                } else {
                    is_bad = true;
                    break;
                }
            }

            if is_bad {
                output_tokens.push(self.unk_token.clone());
            } else {
                output_tokens.extend(sub_tokens);
            }
        }

        output_tokens
    }

    fn encode(&self, text: &str) -> Vec<i32> {
        let re = Regex::new(r"[^\s]+").unwrap();
        let text = text.nfkc().collect::<String>();
        let mut output_ids = Vec::new();

        for token in re.find_iter(&text) {
            let chars: Vec<char> = token.as_str().chars().collect();
            if chars.len() > self.max_input_chars_per_word {
                output_ids.push(self.unk_token_id);
                continue;
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
                    sub_tokens.push(token_id);
                    start += if start == 0 { len } else { len - 2 };
                } else {
                    is_bad = true;
                    break;
                }
            }

            if is_bad {
                output_ids.push(self.unk_token_id);
            } else {
                output_ids.extend(sub_tokens);
            }
        }

        output_ids
    }

    fn decode(&self, ids: Vec<i32>) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab_lookup.get(&id))
            .map(|t| t.replace("##", ""))
            .collect::<String>()
    }
}

#[pymodule]
fn wordpiece_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WordPieceTokenizer>()?;
    Ok(())
}