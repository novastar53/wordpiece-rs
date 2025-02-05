use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use regex::Regex;

#[pyclass]
struct WordPieceTokenizer {
    vocab: HashMap<String, i32>,
    unk_token: String,
    max_input_chars_per_word: usize,
}

#[pymethods]
impl WordPieceTokenizer {
    #[new]
    fn new(vocab: &PyDict, unk_token: Option<&str>, max_input_chars_per_word: Option<usize>) -> Self {
        let mut vocab_map = HashMap::new();
        for (k, v) in vocab.iter() {
            let key = k.extract::<String>().unwrap();
            let value = v.extract::<i32>().unwrap();
            vocab_map.insert(key, value);
        }

        WordPieceTokenizer {
            vocab: vocab_map,
            unk_token: unk_token.unwrap_or("[UNK]").to_string(),
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

            let mut is_bad = false;
            let mut start = 0;
            let mut sub_tokens = Vec::new();

            while start < chars.len() {
                let mut end = chars.len();
                let mut cur_substr = None;

                while start < end {
                    let substr: String = chars[start..end].iter().collect();
                    let substr_to_check = if start == 0 {
                        substr.clone()
                    } else {
                        format!("##{}",substr)
                    };

                    if self.vocab.contains_key(&substr_to_check) {
                        cur_substr = Some(substr_to_check);
                        break;
                    }
                    end -= 1;
                }

                if cur_substr.is_none() {
                    is_bad = true;
                    break;
                }

                sub_tokens.push(cur_substr.unwrap());
                start = end;
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
        self.tokenize(text)
            .iter()
            .map(|token| *self.vocab.get(token).unwrap_or(&-1))
            .collect()
    }

    fn decode(&self, ids: Vec<i32>) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| {
                self.vocab
                    .iter()
                    .find(|(_, &v)| v == id)
                    .map(|(k, _)| k.clone())
            })
            .collect();

        tokens
            .iter()
            .map(|t| t.replace("##", ""))
            .collect::<Vec<String>>()
            .join("")
    }
}

#[pymodule]
fn wordpiece_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WordPieceTokenizer>()?;
    Ok(())
}