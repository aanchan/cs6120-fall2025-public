#!/usr/bin/env python3
"""
Classroom Demo: Static vs Dynamic Embeddings
=============================================

This demo shows why we need attention by comparing:
1. Static embeddings (Word2Vec-style) - same vector regardless of context
2. Dynamic embeddings (Attention-based) - context-dependent vectors

Uses existing visualization code from week6_attention/attention.py

Usage:
    python demo_static_vs_dynamic.py --mask causal     # GPT-style (default)
    python demo_static_vs_dynamic.py --mask bidirectional  # BERT-style
"""

import torch
import torch.nn as nn
import numpy as np
import random
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from week6_attention.attention import (
    AttentionLanguageModel,
    train_attention_model,
    visualize_attention
)
from week1_tokenization.tokenizer import CustomTokenizer


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_corpus():
    """Create corpus with clear 'bank' contexts."""
    return [
        # River bank contexts
        "the river bank was flooded yesterday .",
        "walk along the river bank slowly .",
        "the river bank is muddy and wet .",
        "fish near the river bank often .",

        # Financial bank contexts
        "deposit money at the bank today .",
        "the bank has my savings account .",
        "the bank manager helped me greatly .",
        "withdraw cash from the bank quickly .",

        # Other varied sentences
        "the cat sat on the mat .",
        "dogs love to play outside .",
        "water flows in the river .",
        "people deposit money in accounts .",
    ]


def demo_static_embeddings():
    """Show static embeddings are identical regardless of context."""
    set_seed(42)  # For reproducible embeddings

    print("\n" + "="*70)
    print("PART 1: STATIC EMBEDDINGS (Word2Vec-style)")
    print("="*70)
    print("\nStatic embeddings = ONE vector per word, regardless of context")

    # Create simple embedding layer
    vocab = ["the", "river", "bank", "money", "has", "was", "flooded", "my"]
    word_to_id = {word: i for i, word in enumerate(vocab)}

    embedding_layer = nn.Embedding(len(vocab), 8)

    # Two sentences with "bank"
    sentence1 = ["the", "river", "bank", "flooded"]
    sentence2 = ["the", "bank", "has", "money"]

    # Encode
    ids1 = torch.tensor([word_to_id[w] for w in sentence1])
    ids2 = torch.tensor([word_to_id[w] for w in sentence2])

    # Get embeddings
    embed1 = embedding_layer(ids1)
    embed2 = embedding_layer(ids2)

    # Extract "bank"
    bank1 = embed1[2]  # position 2 in sentence1
    bank2 = embed2[1]  # position 1 in sentence2

    print(f"\nSentence 1: '{' '.join(sentence1)}'")
    print(f"  'bank' embedding: {bank1[:4].detach().numpy()}")

    print(f"\nSentence 2: '{' '.join(sentence2)}'")
    print(f"  'bank' embedding: {bank2[:4].detach().numpy()}")

    # Check if identical
    diff = torch.sum(torch.abs(bank1 - bank2)).item()

    print(f"\n❌ DIFFERENCE: {diff:.10f}")
    print("   ↳ ZERO! Same vector regardless of context")
    print("   ↳ This is the limitation of Word2Vec")


def train_model_with_mask_type(tokenizer, corpus, use_causal_mask=True):
    """Train a model with specified mask type."""
    # Create model with modified forward to accept mask_type
    model = AttentionLanguageModel(
        vocab_size=len(tokenizer.word_to_id),
        embed_dim=32,
        num_heads=2,
        num_layers=2
    )

    # Store mask type preference
    model.use_causal_mask = use_causal_mask

    # Monkey-patch the forward method to use our mask preference
    original_forward = model.forward

    def modified_forward(x, return_attention=False):
        seq_len = x.size(1)

        # Embedding and positional encoding
        x_embedded = model.embedding(x)
        x_pos = model.pos_encoding(x_embedded)
        x_dropped = model.dropout(x_pos)

        # Create mask based on preference
        if model.use_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.ones(seq_len, seq_len).unsqueeze(0).unsqueeze(0)

        mask = mask.to(x.device)

        attention_weights_list = []
        x_current = x_dropped

        # Apply attention layers
        for attention, layer_norm in zip(model.attention_layers, model.layer_norms):
            if return_attention:
                attention_out, attention_weights = attention(x_current, mask, return_attention=True)
                attention_weights_list.append(attention_weights)
            else:
                attention_out = attention(x_current, mask)

            # Residual connection and layer norm
            x_current = layer_norm(x_current + attention_out)

        # Output projection
        output = model.output_linear(x_current)

        if return_attention:
            return output, attention_weights_list
        return output

    model.forward = modified_forward

    # Train
    model = train_attention_model(model, tokenizer, corpus, epochs=100, lr=0.003)

    return model


def demo_with_trained_model(use_causal_mask=True):
    """Train model with specified mask type."""
    set_seed(42)  # For reproducible training

    mask_name = "CAUSAL (GPT-style)" if use_causal_mask else "BIDIRECTIONAL (BERT-style)"
    print("\n" + "="*70)
    print(f"TRAINING {mask_name} MODEL")
    print("="*70)

    corpus = create_corpus()

    # Create tokenizer
    tokenizer = CustomTokenizer(min_freq=1)
    tokenizer.build_vocab(corpus)

    # Train model
    print(f"\nTraining on {len(corpus)} sentences...")
    if use_causal_mask:
        print("• Causal mask: tokens can only see past/current (triangular heatmap)")
        print("• Like GPT: for language modeling / generation")
    else:
        print("• No mask: tokens can see all positions (full heatmap)")
        print("• Like BERT: for understanding / classification")
    print()

    model = train_model_with_mask_type(tokenizer, corpus, use_causal_mask=use_causal_mask)

    return model, tokenizer


def demo_dynamic_embeddings(model, tokenizer):
    """Show that attention creates context-dependent embeddings."""
    print("\n" + "="*70)
    print("PART 2: DYNAMIC EMBEDDINGS (Attention-based)")
    print("="*70)
    print("\nDynamic embeddings = DIFFERENT vector depending on context")

    sentence1 = "the river bank was flooded"
    sentence2 = "the bank has my money"

    # Tokenize
    tokens1 = tokenizer.tokenize(sentence1)
    tokens2 = tokenizer.tokenize(sentence2)

    # Encode
    ids1 = torch.tensor([[tokenizer.word_to_id.get(t, 1) for t in tokens1]])
    ids2 = torch.tensor([[tokenizer.word_to_id.get(t, 1) for t in tokens2]])

    # Get embeddings after attention
    with torch.no_grad():
        # Get static embeddings
        static1 = model.embedding(ids1[0])
        static2 = model.embedding(ids2[0])

        # Apply attention (creates dynamic embeddings)
        dynamic1 = model.pos_encoding(static1.unsqueeze(0))
        dynamic2 = model.pos_encoding(static2.unsqueeze(0))

        dynamic1 = model.attention_layers[0](dynamic1)
        dynamic2 = model.attention_layers[0](dynamic2)

    # Find "bank" positions
    bank_pos1 = tokens1.index("bank")
    bank_pos2 = tokens2.index("bank")

    # Extract "bank" embeddings
    bank_dynamic1 = dynamic1[0, bank_pos1]
    bank_dynamic2 = dynamic2[0, bank_pos2]

    print(f"\nSentence 1: '{sentence1}'")
    print(f"  'bank' embedding: {bank_dynamic1[:4].detach().numpy()}")

    print(f"\nSentence 2: '{sentence2}'")
    print(f"  'bank' embedding: {bank_dynamic2[:4].detach().numpy()}")

    diff = torch.sum(torch.abs(bank_dynamic1 - bank_dynamic2)).item()

    print(f"\n✅ DIFFERENCE: {diff:.6f}")
    print("   ↳ NON-ZERO! Different vectors in different contexts!")
    print("   ↳ Attention creates context-aware representations")


def main():
    """Run complete demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Static vs Dynamic Embeddings Demo")
    parser.add_argument(
        "--mask",
        type=str,
        choices=["causal", "bidirectional"],
        default="causal",
        help="Attention mask type: 'causal' (GPT-style, default) or 'bidirectional' (BERT-style)"
    )
    args = parser.parse_args()

    use_causal_mask = (args.mask == "causal")
    mask_desc = "CAUSAL (GPT-style)" if use_causal_mask else "BIDIRECTIONAL (BERT-style)"

    print("\n🎯 " + "="*66 + " 🎯")
    print("   DEMO: Static vs Dynamic Embeddings")
    print(f"   Attention Type: {mask_desc}")
    print("   " + "="*68)

    # Part 1: Show static embeddings problem
    demo_static_embeddings()

    input("\n⏸️  Press Enter to train model and continue...")

    # Train model
    model, tokenizer = demo_with_trained_model(use_causal_mask=use_causal_mask)

    input("\n⏸️  Press Enter to see dynamic embeddings...")

    # Part 2: Show dynamic embeddings solution
    demo_dynamic_embeddings(model, tokenizer)

    input("\n⏸️  Press Enter to visualize attention patterns...")

    # Part 3: Use existing visualization code!
    print("\n" + "="*70)
    print(f"PART 3: VISUALIZING ATTENTION ({mask_desc})")
    print("="*70)
    print("\nNow let's use the visualization functions from attention.py...")
    print("This will show you HOW attention creates different embeddings.\n")

    # Visualize attention for both sentences
    test_sentences = [
        "the river bank was flooded",
        "the bank has my money"
    ]

    # First, show general instructions
    print("\n    HOW TO READ THE HEATMAPS:")
    print("    • You'll see 4 heatmaps per sentence (2 layers × 2 heads)")
    print("    • Find the 'bank' row on the Y-axis")
    print("    • DARK BLUE = high attention, Light/white = low attention")

    if use_causal_mask:
        print("\n    CAUSAL MASK (GPT-style):")
        print("    • Notice the TRIANGULAR pattern:")
        print("      - Lower left: can attend to current + past tokens")
        print("      - Upper right: MASKED (can't see future)")
        print("    • This is for next-word prediction (language modeling)")
    else:
        print("\n    BIDIRECTIONAL (BERT-style):")
        print("    • Notice the FULL MATRIX pattern:")
        print("      - All positions can attend to all positions")
        print("      - No masking - can see past, present, and future")
        print("    • This is for understanding/encoding tasks")

    print("    • Focus on LAYER 2, HEAD 1 to see semantic patterns")
    print()

    # Sentence 1
    print(f"\n📊 Sentence 1: '{test_sentences[0]}'")
    print("    In Layer 2, Head 1: Look at the 'bank' row")
    if use_causal_mask:
        print("    → bank → river will be DARK BLUE (~0.86)")
        print("    → bank can only see 'the', 'river', and itself")
    else:
        print("    → bank can attend to ALL words (including future 'was', 'flooded')")
        print("    → Look for contextual patterns across the full row")
    print()
    visualize_attention(model, test_sentences[0], tokenizer)

    input("\n⏸️  Press Enter to see second sentence...")

    # Sentence 2
    print(f"\n📊 Sentence 2: '{test_sentences[1]}'")
    print("    In Layer 2, Head 1: Look at the 'bank' row")
    if use_causal_mask:
        print("    → bank → bank (self) will be darkest (~0.78)")
        print("    → No 'river' in visible context")
    else:
        print("    → bank can see ALL words including 'has', 'my', 'money'")
        print("    → Compare attention to future words vs Sentence 1")
    print("    → Compare this to Sentence 1 - DIFFERENT attention pattern!")
    print()
    visualize_attention(model, test_sentences[1], tokenizer)

    # Summary
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print(f"""
1️⃣  STATIC EMBEDDINGS (Word2Vec):
   • One vector per word type - always identical
   • Cannot distinguish different meanings

2️⃣  DYNAMIC EMBEDDINGS (Attention):
   • Different vector per occurrence - context-aware
   • "bank" gets different embeddings based on neighbors

3️⃣  HOW ATTENTION WORKS:
   • Compute attention weights (who to focus on?)
   • Take weighted combination of neighbor embeddings
   • Result: Context-dependent representation

4️⃣  MASK TYPE YOU SAW: {mask_desc}
   {"• Causal: Can only see past/current (triangular heatmap)" if use_causal_mask else "• Bidirectional: Can see all tokens (full heatmap)"}
   {"• Used in: GPT, language modeling, generation" if use_causal_mask else "• Used in: BERT, understanding, classification"}
   {"• Try --mask bidirectional to compare!" if use_causal_mask else "• Try --mask causal to compare!"}

5️⃣  VISUALIZATION SHOWED:
   • In "river bank" - 'bank' attends to contextual words
   • In "money bank" - different attention pattern
   • This is why the embeddings are different!

🎓 This is the CORE mechanism behind transformers and modern LLMs!
   Run with different --mask options to see both GPT and BERT styles!
    """)


if __name__ == "__main__":
    main()
