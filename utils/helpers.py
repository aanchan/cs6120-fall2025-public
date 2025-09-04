"""
Utility functions for the NLP course project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json
import pickle

def save_model(model, filepath: str):
    """Save a PyTorch model."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath: str):
    """Load a PyTorch model."""
    model.load_state_dict(torch.load(filepath))
    return model

def save_tokenizer(tokenizer, filepath: str):
    """Save tokenizer vocabulary."""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'word_to_id': tokenizer.word_to_id,
            'id_to_word': tokenizer.id_to_word,
            'word_freq': dict(tokenizer.word_freq)
        }, f)

def load_tokenizer(tokenizer, filepath: str):
    """Load tokenizer vocabulary."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        tokenizer.word_to_id = data['word_to_id']
        tokenizer.id_to_word = data['id_to_word']
        tokenizer.word_freq = data['word_freq']

def plot_training_curves(train_losses, val_losses, title="Training Curves"):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_perplexity(model, test_loader):
    """Calculate perplexity on test set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                input_ids, labels = batch
                logits, loss = model(input_ids, labels)
            else:
                # Handle different batch formats
                logits, loss = model(batch)
            
            total_loss += loss.item()
            total_tokens += 1
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def generate_text_samples(model, tokenizer, prompts: List[str], max_length: int = 50):
    """Generate text samples for given prompts."""
    model.eval()
    samples = {}
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids])
            
            # Generate
            generated_ids = model.generate(
                input_tensor,
                max_new_tokens=max_length,
                temperature=1.0
            )
            
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            samples[prompt] = generated_text
    
    return samples

def evaluate_model_performance(model, test_loader, tokenizer):
    """Evaluate model performance with multiple metrics."""
    model.eval()
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, test_loader)
    
    # Generate sample texts
    test_prompts = ["The cat", "Dogs love", "In the garden"]
    samples = generate_text_samples(model, tokenizer, test_prompts)
    
    results = {
        'perplexity': perplexity,
        'samples': samples
    }
    
    return results

def create_experiment_log(experiment_name: str, config: Dict[str, Any], results: Dict[str, Any]):
    """Create a log entry for an experiment."""
    log_entry = {
        'experiment_name': experiment_name,
        'config': config,
        'results': results,
        'timestamp': str(np.datetime64('now'))
    }
    
    return log_entry

def save_experiment_log(log_entry, filepath: str):
    """Save experiment log to file."""
    with open(filepath, 'w') as f:
        json.dump(log_entry, f, indent=2)

def load_experiment_log(filepath: str):
    """Load experiment log from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_models(model_results: Dict[str, Dict]):
    """Compare multiple models and their results."""
    print("Model Comparison:")
    print("=" * 50)
    
    for model_name, results in model_results.items():
        print(f"\n{model_name}:")
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print("  Sample generations:")
        for prompt, text in results['samples'].items():
            print(f"    '{prompt}' -> '{text}'")

def create_model_summary(model):
    """Create a summary of model architecture and parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': str(model)
    }
    
    return summary

def print_model_summary(model):
    """Print a formatted model summary."""
    summary = create_model_summary(model)
    
    print("Model Summary:")
    print("=" * 30)
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Model Size: {summary['model_size_mb']:.2f} MB")
    print("\nArchitecture:")
    print(summary['architecture'])

# Data utilities
def load_text_file(filepath: str) -> List[str]:
    """Load text from file and split into lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def save_text_file(lines: List[str], filepath: str):
    """Save lines to text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def split_data(data: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split data into train/validation/test sets."""
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

# Visualization utilities
def plot_attention_weights(attention_weights, tokens, title="Attention Weights"):
    """Plot attention weights as a heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()
    plt.show()

def plot_embedding_visualization(embeddings, words, method='tsne'):
    """Visualize word embeddings in 2D."""
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce to 2D
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Add word labels
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 2), textcoords='offset points')
    
    plt.title(f"Word Embeddings ({method.upper()})")
    plt.show()

# Training utilities
def create_learning_rate_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create a learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def gradient_clipping(model, max_norm: float = 1.0):
    """Apply gradient clipping to model parameters."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def early_stopping(val_losses: List[float], patience: int = 5) -> bool:
    """Check if training should stop early."""
    if len(val_losses) < patience:
        return False
    
    recent_losses = val_losses[-patience:]
    return all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses)))

# Evaluation utilities
def calculate_bleu_score(references: List[List[str]], candidates: List[List[str]]) -> float:
    """Calculate BLEU score for text generation evaluation."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    smoothing = SmoothingFunction().method1
    scores = []
    
    for ref, cand in zip(references, candidates):
        score = sentence_bleu([ref], cand, smoothing_function=smoothing)
        scores.append(score)
    
    return np.mean(scores)

def calculate_diversity_score(generated_texts: List[str]) -> float:
    """Calculate diversity of generated texts using unique n-grams."""
    from collections import Counter
    
    all_ngrams = []
    for text in generated_texts:
        words = text.split()
        # Create bigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        all_ngrams.extend(bigrams)
    
    ngram_counts = Counter(all_ngrams)
    total_ngrams = len(all_ngrams)
    unique_ngrams = len(ngram_counts)
    
    diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    return diversity 