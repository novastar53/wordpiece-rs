use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use unicode_normalization::UnicodeNormalization;
use regex::{Regex, RegexBuilder};

#[derive(Debug, Clone)]
struct Symbol {
    text: String,
    count: usize,
    score: f64,
}

impl Symbol {
    fn new(text: String, count: usize) -> Self {
        Symbol {
            text,
            count,
            score: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct WordPieceTrainer {
    vocab_size: usize,
    min_frequency: usize,
    special_tokens: Vec<String>,
    basic_tokenizer: Regex,
    punctuation: Regex,
    chinese_chars: Regex,
    strip_accents: bool,
    lowercase: bool,
}

impl WordPieceTrainer {
    pub fn new(
        vocab_size: usize,
        min_frequency: usize,
        special_tokens: Vec<String>,
        strip_accents: bool,
        lowercase: bool,
    ) -> Self {
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

        WordPieceTrainer {
            vocab_size,
            min_frequency,
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

    fn strip_accents_if_needed(&self, text: &str) -> String {
        if !self.strip_accents {
            return text.to_string();
        }

        text.nfd()
            .filter(|&c| !c.is_ascii_punctuation() && !c.is_ascii_control())
            .collect::<String>()
    }

    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let text = self.clean_text(text);
        
        for mat in self.basic_tokenizer.find_iter(&text) {
            let mut token_text = mat.as_str().trim().to_string();
            
            // Handle casing
            if self.lowercase {
                token_text = token_text.to_lowercase();
            }
            
            // Handle accents
            token_text = self.strip_accents_if_needed(&token_text);
            
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
            
            tokens.extend(char_tokens);
        }
        
        tokens
    }

    fn get_initial_symbols(&self, texts: &[String]) -> HashMap<String, Symbol> {
        let mut char_counts: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        // First pass: count characters and words
        for text in texts {
            let tokens = self.basic_tokenize(text);
            for token in tokens {
                *word_counts.entry(token.clone()).or_insert(0) += 1;
                for c in token.chars() {
                    *char_counts.entry(c.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Create initial symbols from characters that appear in frequent words
        let mut symbols: HashMap<String, Symbol> = HashMap::new();
        
        // Add special tokens first
        for token in &self.special_tokens {
            symbols.insert(
                token.clone(),
                Symbol::new(token.clone(), word_counts.get(token).copied().unwrap_or(0)),
            );
        }

        // Add characters from words that meet minimum frequency
        for (word, &count) in &word_counts {
            if count >= self.min_frequency {
                for c in word.chars() {
                    let c_str = c.to_string();
                    if !symbols.contains_key(&c_str) {
                        symbols.insert(
                            c_str.clone(),
                            Symbol::new(c_str, *char_counts.get(&c.to_string()).unwrap()),
                        );
                    }
                }
            }
        }

        symbols
    }

    fn compute_pair_scores(
        &self,
        texts: &[String],
        symbols: &HashMap<String, Symbol>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let symbol_set: HashSet<_> = symbols.keys().collect();

        for text in texts {
            let tokens = self.basic_tokenize(text);
            for token in tokens {
                if token.len() <= 1 {
                    continue;
                }

                let mut current_symbols = Vec::new();
                let mut current = String::new();

                // Split word into known symbols
                for c in token.chars() {
                    current.push(c);
                    if symbol_set.contains(&current) {
                        current_symbols.push(current.clone());
                        current.clear();
                    }
                }

                // Count adjacent pairs
                for pair in current_symbols.windows(2) {
                    if let [first, second] = pair {
                        *pair_counts.entry((first.clone(), second.clone())).or_insert(0) += 1;
                    }
                }
            }
        }

        pair_counts
    }

    fn merge_symbols(
        &self,
        symbols: &mut HashMap<String, Symbol>,
        pair_counts: &HashMap<(String, String), usize>,
    ) -> Option<(String, String)> {
        // Find the best pair to merge
        let mut best_pair = None;
        let mut best_score = 0.0;

        for ((first, second), &count) in pair_counts {
            if count < self.min_frequency {
                continue;
            }

            // Skip if either symbol is not in vocabulary (might have been merged)
            if !symbols.contains_key(first) || !symbols.contains_key(second) {
                continue;
            }

            // Compute score using frequency-based heuristic
            let score = count as f64 / 
                (symbols[first].count as f64 * symbols[second].count as f64);

            match score.partial_cmp(&best_score) {
                Some(Ordering::Greater) => {
                    best_score = score;
                    best_pair = Some((first.clone(), second.clone()));
                }
                _ => {}
            }
        }

        // If we found a pair to merge, create new symbol
        if let Some((first, second)) = best_pair.clone() {
            let merged = format!("{}{}", first, second);
            let count = pair_counts[&(first.clone(), second.clone())];
            
            symbols.insert(
                merged.clone(),
                Symbol::new(merged, count),
            );
        }

        best_pair
    }

    pub fn train(&self, texts: &[String]) -> HashMap<String, i32> {
        let mut symbols = self.get_initial_symbols(texts);
        let mut vocab: HashMap<String, i32> = HashMap::new();
        let mut next_id = 0;

        // Add special tokens first
        for token in &self.special_tokens {
            vocab.insert(token.clone(), next_id);
            next_id += 1;
        }

        while vocab.len() < self.vocab_size {
            // Compute pair frequencies
            let pair_counts = self.compute_pair_scores(texts, &symbols);
            
            // Find and merge best pair
            match self.merge_symbols(&mut symbols, &pair_counts) {
                Some((first, second)) => {
                    let merged = format!("{}{}", first, second);
                    if !vocab.contains_key(&merged) {
                        vocab.insert(merged, next_id);
                        next_id += 1;
                    }
                }
                None => break, // No more pairs to merge
            }
        }

        // Add remaining single-character symbols if space permits
        for (symbol, _) in symbols.iter() {
            if vocab.len() >= self.vocab_size {
                break;
            }
            if !vocab.contains_key(symbol) {
                vocab.insert(symbol.clone(), next_id);
                next_id += 1;
            }
        }

        vocab
    }
}