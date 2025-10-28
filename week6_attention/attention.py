import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SelfAttention(nn.Module):
    """Scaled dot-product self-attention."""
    
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: input tensor (batch_size, seq_len, embed_dim)
            mask: attention mask (batch_size, seq_len, seq_len)
            return_attention: whether to return attention weights
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        Q = self.q_linear(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        # scores: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # attention_output: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final linear projection
        output = self.out_linear(attention_output)
        
        if return_attention:
            return output, attention_weights
        return output

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            -(np.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AttentionLanguageModel(nn.Module):
    """Language model with self-attention."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        self.attention_layers = nn.ModuleList([
            SelfAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        self.output_linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, return_attention=False):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(x.device)
        
        attention_weights_list = []
        
        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            if return_attention:
                attention_out, attention_weights = attention(x, mask, return_attention=True)
                attention_weights_list.append(attention_weights)
            else:
                attention_out = attention(x, mask)
            
            # Residual connection and layer norm
            x = layer_norm(x + attention_out)
        
        # Output projection
        output = self.output_linear(x)
        
        if return_attention:
            return output, attention_weights_list
        return output

def visualize_attention(model, text, tokenizer):
    """Visualize attention weights for given text."""
    model.eval()
    
    # Encode text
    tokens = tokenizer.encode(text)
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    
    # Get model output with attention weights
    with torch.no_grad():
        output, attention_weights_list = model(input_tensor, return_attention=True)
    
    # Get tokens for visualization
    tokens_text = [tokenizer.id_to_word.get(t, '<UNK>') for t in tokens]
    
    # Visualize attention for each layer and head
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = attention_weights[0].cpu().numpy()  # Remove batch dimension
        
        num_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))
        
        if num_heads == 1:
            axes = [axes]
        
        for head_idx, ax in enumerate(axes):
            sns.heatmap(
                attention_weights[head_idx],
                xticklabels=tokens_text,
                yticklabels=tokens_text,
                cmap='Blues',
                ax=ax,
                cbar=True
            )
            ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
            ax.set_xlabel('Keys')
            ax.set_ylabel('Queries')
        
        plt.tight_layout()
        plt.show()

def analyze_attention_patterns(model, tokenizer, test_texts):
    """Analyze attention patterns across different texts."""
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)
    
    for text in test_texts:
        print(f"\nAnalyzing: '{text}'")
        visualize_attention(model, text, tokenizer)

def compare_attention_heads(model, tokenizer, text):
    """Compare attention patterns across different heads."""
    model.eval()
    
    tokens = tokenizer.encode(text)
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    tokens_text = [tokenizer.id_to_word.get(t, '<UNK>') for t in tokens]
    
    with torch.no_grad():
        output, attention_weights_list = model(input_tensor, return_attention=True)
    
    # Analyze first layer attention
    first_layer_attention = attention_weights_list[0][0].cpu().numpy()
    num_heads = first_layer_attention.shape[0]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            first_layer_attention[head_idx],
            xticklabels=tokens_text,
            yticklabels=tokens_text,
            cmap='Blues',
            ax=ax,
            cbar=True
        )
        ax.set_title(f'Head {head_idx+1}')
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')
    
    plt.suptitle(f'Attention Patterns: "{text}"', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Analyze attention statistics
    print(f"\nAttention Statistics for: '{text}'")
    print("-" * 40)
    
    for head_idx in range(num_heads):
        attention_matrix = first_layer_attention[head_idx]
        max_attention = np.max(attention_matrix)
        min_attention = np.min(attention_matrix)
        mean_attention = np.mean(attention_matrix)
        
        print(f"Head {head_idx+1}:")
        print(f"  Max attention: {max_attention:.3f}")
        print(f"  Min attention: {min_attention:.3f}")
        print(f"  Mean attention: {mean_attention:.3f}")

def visualize_positional_encoding(embed_dim=64, max_len=20):
    """Visualize positional encoding patterns."""
    pe = PositionalEncoding(embed_dim, max_len)
    
    # Create dummy input
    x = torch.randn(1, max_len, embed_dim)
    pe_output = pe(x)
    
    # Visualize positional encoding
    plt.figure(figsize=(15, 5))
    
    # Plot first few dimensions
    plt.subplot(1, 3, 1)
    for i in range(5):
        plt.plot(pe_output[0, :, i].numpy(), label=f'Dim {i}')
    plt.title('First 5 Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot middle dimensions
    plt.subplot(1, 3, 2)
    for i in range(embed_dim//2-2, embed_dim//2+3):
        plt.plot(pe_output[0, :, i].numpy(), label=f'Dim {i}')
    plt.title('Middle Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot last few dimensions
    plt.subplot(1, 3, 3)
    for i in range(embed_dim-5, embed_dim):
        plt.plot(pe_output[0, :, i].numpy(), label=f'Dim {i}')
    plt.title('Last 5 Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Positional Encoding Visualization', fontsize=16)
    plt.tight_layout()
    plt.show()

def train_attention_model(model, tokenizer, corpus, epochs=50, lr=0.001):
    """Train the attention model on the corpus."""
    print(f"\nTraining attention model for {epochs} epochs...")
    
    # Create training data
    encoded_corpus = [tokenizer.encode(text) for text in corpus]
    
    # Create simple training pairs (context -> next token)
    training_pairs = []
    for text in encoded_corpus:
        for i in range(1, len(text)):
            context = text[:i]
            target = text[i]
            training_pairs.append((context, target))
    
    # Convert to tensors
    max_len = max(len(pair[0]) for pair in training_pairs)
    X = []
    y = []
    
    for context, target in training_pairs:
        # Pad context to max_len
        padded_context = context + [0] * (max_len - len(context))
        X.append(padded_context)
        y.append(target)
    
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        # Take the last position output for each sequence
        last_outputs = outputs[:, -1, :]  # (batch_size, vocab_size)
        
        loss = criterion(last_outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    return model

def analyze_trained_attention(model, tokenizer, test_sentences):
    """Analyze attention patterns in the trained model."""
    print("\n" + "="*60)
    print("ANALYZING TRAINED ATTENTION PATTERNS")
    print("="*60)
    print("\nWHAT TO LOOK FOR IN ATTENTION PLOTS:")
    print("• Bright squares = high attention weights (close to 1.0)")
    print("• Dark squares = low attention weights (close to 0.0)")
    print("• Look for meaningful linguistic relationships!")
    print("• Different heads may specialize in different types of relationships")
    print("="*60)
    
    model.eval()
    
    for sentence in test_sentences:
        print(f"\nAnalyzing: '{sentence}'")
        
        # Provide specific guidance for each sentence
        if "cat sat" in sentence.lower():
            print("Look for: 'sat' attending to 'cat' (subject-verb relationship)")
        elif "it was dirty" in sentence.lower():
            print("Look for: 'it' attending to 'dog' or 'bath' (pronoun resolution)")
        elif "chased ran" in sentence.lower():
            print("Look for: 'ran' attending to 'cat' (subject across relative clause)")
        elif "red car" in sentence.lower():
            print("Look for: 'red' attending to 'car' (modifier-noun relationship)")
        elif "she waved" in sentence.lower():
            print("Look for: 'she' attending to 'Alice' (pronoun-antecedent)")
        
        # Encode sentence
        tokens = tokenizer.encode(sentence)
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Get attention weights
        with torch.no_grad():
            output, attention_weights_list = model(input_tensor, return_attention=True)
        
        # Get token texts
        token_texts = [tokenizer.id_to_word.get(t, '<UNK>') for t in tokens]
        
        # Visualize attention for each layer
        for layer_idx, attention_weights in enumerate(attention_weights_list):
            attention_weights = attention_weights[0].cpu().numpy()  # Remove batch dimension
            
            num_heads = attention_weights.shape[0]
            fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))
            
            if num_heads == 1:
                axes = [axes]
            
            for head_idx, ax in enumerate(axes):
                sns.heatmap(
                    attention_weights[head_idx],
                    xticklabels=token_texts,
                    yticklabels=token_texts,
                    cmap='Blues',
                    ax=ax,
                    cbar=True
                )
                
                # Add specific guidance based on head and layer
                if layer_idx == 0:
                    if head_idx == 0:
                        ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}\n(Local/Syntactic Relationships)')
                    else:
                        ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}\n(Semantic Relationships)')
                else:
                    if head_idx == 0:
                        ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}\n(Complex/Long-distance)')
                    else:
                        ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}\n(Specialized Patterns)')
                
                ax.set_xlabel('Keys (What each word attends TO)')
                ax.set_ylabel('Queries (What each word attends FROM)')
            
            plt.tight_layout()
            plt.show()
            
            # Analyze attention patterns with specific guidance
            print(f"\nLayer {layer_idx+1} Analysis:")
            for head_idx in range(num_heads):
                attention_matrix = attention_weights[head_idx]
                
                # Find strongest attention patterns
                max_attention = np.max(attention_matrix)
                max_positions = np.where(attention_matrix == max_attention)
                
                print(f"  Head {head_idx+1}:")
                print(f"    Max attention: {max_attention:.3f}")
                if len(max_positions[0]) > 0:
                    for i, j in zip(max_positions[0], max_positions[1]):
                        if i != j:  # Skip self-attention
                            print(f"    '{token_texts[i]}' -> '{token_texts[j]}' (attention: {max_attention:.3f})")
                
                # Look for specific linguistic patterns
                print(f"    Looking for patterns:")
                if "cat" in token_texts and "sat" in token_texts:
                    cat_idx = token_texts.index("cat")
                    sat_idx = token_texts.index("sat")
                    if sat_idx < len(attention_matrix) and cat_idx < len(attention_matrix[0]):
                        sat_to_cat = attention_matrix[sat_idx, cat_idx]
                        print(f"      'sat' -> 'cat': {sat_to_cat:.3f}")
                
                if "it" in token_texts:
                    it_idx = token_texts.index("it")
                    if it_idx < len(attention_matrix):
                        # Find what "it" attends to most
                        it_attention = attention_matrix[it_idx]
                        max_attended = np.argmax(it_attention)
                        if max_attended != it_idx:
                            print(f"      'it' -> '{token_texts[max_attended]}': {it_attention[max_attended]:.3f}")

# Test script
if __name__ == "__main__":
    # Test self-attention mechanism
    embed_dim = 64
    num_heads = 2
    seq_len = 10
    batch_size = 2
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create attention module
    attention = SelfAttention(embed_dim, num_heads)
    
    # Forward pass
    output = attention(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    from week1_tokenization.tokenizer import CustomTokenizer
    
    # Illustrative corpus for attention visualization
    corpus = [
        # Simple sentences for baseline
        "The cat sat on the mat .",
        "The dog chased the cat .",
        
        # Sentences with clear dependencies
        "The cat that the dog chased ran away quickly .",
        "She gave her dog a bath because it was dirty .",
        "Alice saw Bob and she waved to him .",
        "The book that you gave me was very interesting .",
        
        # Longer, more complex sentences
        "The red car that John bought yesterday is parked in the garage near the house .",
        "When the weather is cold, people wear warm clothes and stay inside their homes .",
        "The teacher explained the difficult concept to the confused students who asked many questions .",
        "The mouse that the cat that the dog chased caught squeaked loudly .",
        "If the dog barks loudly, the cat hides under the table and waits quietly .",
        
        # Sentences with multiple relationships
        "John told Mary that he would help her with the project because she was struggling .",
        "The beautiful flowers that bloom in spring attract many bees and butterflies .",
        "The old man who lives in the house on the hill walks his dog every morning .",
        "Students who study hard and practice regularly usually perform well on exams .",
        "The chef who prepared the delicious meal received many compliments from satisfied customers ."
    ]
    
    # Create tokenizer
    tokenizer = CustomTokenizer(min_freq=1)
    tokenizer.build_vocab(corpus)
    
    # Create model
    model = AttentionLanguageModel(
        vocab_size=len(tokenizer.word_to_id),
        embed_dim=64,
        num_heads=2,
        num_layers=2
    )
    
    # Train the model
    model = train_attention_model(model, tokenizer, corpus, epochs=50, lr=0.001)
    
    # Analyze trained attention patterns
    test_sentences = [
        "The cat sat on the mat .",
        "The cat that the dog chased ran away .",
        "She gave her dog a bath because it was dirty .",
        "Alice saw Bob and she waved to him .",
        "The red car that John bought yesterday is parked in the garage ."
    ]
    
    analyze_trained_attention(model, tokenizer, test_sentences) 