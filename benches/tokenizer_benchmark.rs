use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fake::{Fake, Faker};
use fake::locales::EN;
use fake::faker::lorem::en::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use wordpiece_rs::WordPieceTokenizer;

fn create_test_vocab() -> HashMap<String, i32> {
    let mut vocab = HashMap::new();
    // Add special tokens
    vocab.insert("[UNK]".to_string(), 0);
    vocab.insert("[CLS]".to_string(), 1);
    vocab.insert("[SEP]".to_string(), 2);
    vocab.insert("[PAD]".to_string(), 3);
    vocab.insert("[MASK]".to_string(), 4);

    // Add common English subwords
    let subwords = vec![
        "the", "##s", "##ing", "##ed", "##ly", "##er", "##est", "un##", "re##",
        "in##", "to", "of", "and", "##a", "##e", "##i", "##o", "##u", "##t",
        "##n", "##s", "##r", "##l", "##d", "##m", "##p", "##c", "##b", "##f",
        "##g", "##h", "##k", "##w", "##y", "##v", "##x", "##z", "##j", "##q",
    ];

    for (i, word) in subwords.into_iter().enumerate() {
        vocab.insert(word.to_string(), (i + 5) as i32);
    }

    vocab
}

fn generate_test_text(size: usize) -> String {
    let paragraphs: Vec<String> = (0..size).map(|_| Paragraph(3..8).fake()).collect();
    paragraphs.join(" ")
}

fn benchmark_tokenization(c: &mut Criterion) {
    let vocab = create_test_vocab();
    let mut group = c.benchmark_group("tokenization");

    // Test different input sizes
    for size in [1, 10, 100, 1000].iter() {
        let text = generate_test_text(*size);
        
        group.bench_with_input(BenchmarkId::new("tokenize", size), &text,
            |b, text| {
                b.iter(|| {
                    let tokenizer = WordPieceTokenizer::new(
                        vocab.clone(),
                        "[UNK]",
                        200,
                        true,
                        true
                    );
                    black_box(tokenizer.tokenize(text));
                });
            }
        );
    }

    group.finish();
}

fn benchmark_encoding(c: &mut Criterion) {
    let vocab = create_test_vocab();
    let mut group = c.benchmark_group("encoding");

    // Test different input sizes
    for size in [1, 10, 100, 1000].iter() {
        let text = generate_test_text(*size);
        
        group.bench_with_input(BenchmarkId::new("encode", size), &text,
            |b, text| {
                b.iter(|| {
                    let tokenizer = WordPieceTokenizer::new(
                        vocab.clone(),
                        "[UNK]",
                        200,
                        true,
                        true
                    );
                    black_box(tokenizer.encode(text));
                });
            }
        );
    }

    group.finish();
}

fn benchmark_decoding(c: &mut Criterion) {
    let vocab = create_test_vocab();
    let mut group = c.benchmark_group("decoding");

    // Test different input sizes
    for size in [1, 10, 100, 1000].iter() {
        let text = generate_test_text(*size);
        let tokenizer = WordPieceTokenizer::new(
            vocab.clone(),
            "[UNK]",
            200,
            true,
            true
        );
        let ids = tokenizer.encode(&text);
        
        group.bench_with_input(BenchmarkId::new("decode", size), &ids,
            |b, ids| {
                b.iter(|| {
                    black_box(tokenizer.decode(ids.clone()));
                });
            }
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_tokenization, benchmark_encoding, benchmark_decoding);
criterion_main!(benches);