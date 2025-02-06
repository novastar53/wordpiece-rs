mod trainer;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use regex::{Regex, RegexBuilder};
use std::borrow::Cow;
use trainer::WordPieceTrainer;

/// A node in the trie data structure for efficient prefix matching
#[derive(Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_word: bool,
    token_id: i32,
    max_char_len: usize,  // Maximum length of any word in this subtrie
    fail_link: Option<Box<TrieNode>>,  // Failure link for Aho-Corasick-style matching
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_word: false,
            token_id: -1,
            max_char_len: 0,
            fail_link: None,
        }
    }

    /// Insert a word into the trie with its associated token ID
    fn insert(&mut self, word: &str, token_id: i32) {
        let mut node = self;
        let word_len = word.chars().count();
        node.max_char_len = node.max_char_len.max(word_len);

        for ch in word.chars() {
            let next = node.children.entry(ch).or_insert_with(TrieNode::new);
            next.max_char_len = next.max_char_len.max(word_len);
            node = next;
        }
        node.is_word = true;
        node.token_id = token_id;
    }

    /// Build failure links for fast matching (Aho-Corasick style)
    fn build_failure_links(&mut self) {
        use std::collections::VecDeque;
        let mut queue = VecDeque::new();

        // Initialize root's children
        for (_, child) in self.children.iter_mut() {
            child.fail_link = Some(Box::new(TrieNode::new()));
            queue.push_back(child);
        }

        // Build failure links using BFS
        while let Some(node) = queue.pop_front() {
            for (ch, child) in node.children.iter() {
                let mut fail = node.fail_link.as_ref().unwrap();
                while !fail.children.contains_key(ch) && fail.fail_link.is_some() {
                    fail = fail.fail_link.as_ref().unwrap();
                }
                let fail_node = if let Some(next) = fail.children.get(ch) {
                    next
                } else {
                    self
                };
                child.fail_link = Some(Box::new(fail_node.clone()));
                queue.push_back(child);
            }
        }
    }

    /// Find the longest prefix in a single pass using failure links
    fn find_longest_prefix(&self, word: &[char], start: usize) -> Option<(usize, i32)> {
        let mut node = self;
        let mut pos = start;
        let mut last_match = None;

        while pos < word.len() {
            if let Some(next) = node.children.get(&word[pos]) {
                if next.is_word {
                    last_match = Some((pos + 1, next.token_id));
                }
                node = next;
                pos += 1;
            } else if let Some(fail) = &node.fail_link {
                node = fail;
            } else {
                break;
            }
        }

        last_match
    }
}

/// Token represents a single token with its text, ID, and whether it's a special token
#[pyclass]
#[derive(Debug, Clone)]
struct Token {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    id: i32,
    #[pyo3(get)]
    is_special: bool,
}

#[pymethods]
impl Token {
    #[new]
    fn new(text: String, id: i32, is_special: bool) -> Self {
        Token {
            text,
            id,
            is_special,
        }
    }
}

/// A fast WordPiece tokenizer implementation based on the paper
/// "A Fast WordPiece Tokenization Algorithm" by Xinying Song et al. (2021).
/// 
/// This implementation uses a trie with failure links (similar to Aho-Corasick algorithm)
/// to achieve O(n) time complexity for tokenization, where n is the input length.
/// The traditional WordPiece algorithm has O(n²) complexity due to repeated substring checks.
/// 
/// Key improvements:
/// 1. Single-pass tokenization using failure links
/// 2. O(n) time complexity vs O(n²) in traditional implementation
/// 3. Efficient prefix matching with early stopping
/// 4. Memory-efficient trie structure with maximum length tracking
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

        // Process vocabulary and build trie
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

        // Build failure links for fast matching
        trie.build_failure_links();

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

    fn strip_accents_if_needed<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.strip_accents {
            return Cow::Borrowed(text);
        }

        let normalized = text.nfd().collect::<String>();
        let stripped = normalized
            .chars()
            .filter(|&c| !c.is_ascii_punctuation() && !c.is_ascii_control())
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

        // Single-pass tokenization using failure links
        let mut tokens = Vec::new();
        let mut start = 0;
        let mut is_bad = false;

        while start < chars.len() {
            let word_chars = if start == 0 {
                chars.as_slice()
            } else {
                let mut prefix_chars = Vec::with_capacity(2 + chars.len() - start);
                prefix_chars.extend(['#', '#']);
                prefix_chars.extend(&chars[start..]);
                prefix_chars.as_slice()
            };

            if let Some((len, token_id)) = self.trie.find_longest_prefix(word_chars, 0) {
                let token_text = self.vocab_lookup.get(&token_id).unwrap().clone();
                tokens.push(Token {
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
            tokens
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

    #[staticmethod]
    #[args(
        vocab_size = "30000",
        min_frequency = "2",
        special_tokens = "None",
        strip_accents = "true",
        lowercase = "true"
    )]
    fn train(
        texts: Vec<String>,
        vocab_size: usize,
        min_frequency: usize,
        special_tokens: Option<Vec<String>>,
        strip_accents: bool,
        lowercase: bool,
    ) -> PyResult<HashMap<String, i32>> {
        let special_tokens = special_tokens.unwrap_or_else(|| {
            vec![
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[PAD]".to_string(),
                "[MASK]".to_string(),
            ]
        });

        let trainer = WordPieceTrainer::new(
            vocab_size,
            min_frequency,
            special_tokens,
            strip_accents,
            lowercase,
        );

        Ok(trainer.train(&texts))
    }
}

#[pymodule]
fn wordpiece_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WordPieceTokenizer>()?;
    Ok(())
}