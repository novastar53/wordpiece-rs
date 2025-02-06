import timeit
import random
import string
from typing import Dict, List
import wordpiece_rs
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_test_vocab() -> Dict[str, int]:
    """Create a test vocabulary with common subwords."""
    vocab = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "[MASK]": 4,
    }
    
    # Add common English subwords
    subwords = [
        "the", "##s", "##ing", "##ed", "##ly", "##er", "##est", "un##", "re##",
        "in##", "to", "of", "and", "##a", "##e", "##i", "##o", "##u", "##t",
        "##n", "##s", "##r", "##l", "##d", "##m", "##p", "##c", "##b", "##f",
        "##g", "##h", "##k", "##w", "##y", "##v", "##x", "##z", "##j", "##q",
    ]
    
    for i, word in enumerate(subwords):
        vocab[word] = i + 5
        
    return vocab

def generate_test_text(size: int) -> str:
    """Generate random text of given size (in words)."""
    words = []
    for _ in range(size):
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    return ' '.join(words)

def benchmark_tokenizers(sizes: List[int], num_runs: int = 5) -> pd.DataFrame:
    """Benchmark different tokenizer implementations."""
    vocab = create_test_vocab()
    
    # Initialize tokenizers
    rust_tokenizer = wordpiece_rs.WordPieceTokenizer(vocab)
    huggingface_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    results = []
    
    for size in sizes:
        text = generate_test_text(size)
        
        # Benchmark Rust implementation
        rust_time = timeit.timeit(
            lambda: rust_tokenizer.tokenize(text),
            number=num_runs
        ) / num_runs
        
        # Benchmark HuggingFace implementation
        hf_time = timeit.timeit(
            lambda: huggingface_tokenizer.tokenize(text),
            number=num_runs
        ) / num_runs
        
        results.append({
            'size': size,
            'implementation': 'Rust (Fast)',
            'time': rust_time
        })
        results.append({
            'size': size,
            'implementation': 'HuggingFace',
            'time': hf_time
        })
    
    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame, output_file: str = 'benchmark_results.png'):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='size',
        y='time',
        hue='implementation',
        marker='o'
    )
    plt.title('WordPiece Tokenizer Performance Comparison')
    plt.xlabel('Input Size (words)')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    # Run benchmarks
    sizes = [10, 100, 1000, 10000]
    results = benchmark_tokenizers(sizes)
    
    # Print results
    print("\nBenchmark Results:")
    print("=================")
    for size in sizes:
        size_results = results[results['size'] == size]
        print(f"\nInput size: {size} words")
        for _, row in size_results.iterrows():
            print(f"{row['implementation']}: {row['time']:.6f} seconds")
    
    # Plot results
    plot_results(results)
    print("\nPlot saved as 'benchmark_results.png'")

if __name__ == '__main__':
    main()