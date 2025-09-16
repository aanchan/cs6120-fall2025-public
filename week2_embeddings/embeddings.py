import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class SkipGramDataset(Dataset):
    """Dataset for skip-gram training."""
    
    def __init__(self, encoded_texts: List[List[int]], window_size: int = 2):
        self.window_size = window_size
        self.pairs = []
        
        # Create skip-gram pairs
        for text in encoded_texts:
            for i, center_word in enumerate(text):
                # Get context words within window
                start = max(0, i - window_size)
                end = min(len(text), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Skip the center word itself
                        self.pairs.append((center_word, text[j]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)

class Word2Vec(nn.Module):
    """Simple Word2Vec implementation."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        # Two embedding matrices: one for center words, one for context
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center_words, context_words):
        # Get embeddings
        center_embeds = self.center_embeddings(center_words)
        context_embeds = self.context_embeddings(context_words)
        
        # Compute dot product
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        
        return scores
    
    def get_embeddings(self):
        """Return the center embeddings (these are typically used)."""
        return self.center_embeddings.weight.detach().cpu().numpy()

def train_word2vec(model, dataset, vocab_size, epochs=100, lr=0.01):
    """Train Word2Vec model."""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for center_words, context_words in dataloader:
            # Forward pass
            scores = model(center_words, context_words)
            
            # Create negative samples (simplified version)
            batch_size = center_words.size(0)
            neg_context = torch.randint(0, vocab_size, (batch_size,))
            neg_scores = model(center_words, neg_context)
            
            # Compute loss (positive samples should have high scores)
            pos_loss = -torch.log(torch.sigmoid(scores)).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores)).mean()
            loss = pos_loss + neg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

def visualize_embeddings(embeddings, word_to_id, words_to_plot=None):
    """Visualize word embeddings using t-SNE."""
    if words_to_plot is None:
        # Plot first 50 words
        words_to_plot = list(word_to_id.keys())[:50]
    
    # Get indices for words to plot
    indices = [word_to_id[word] for word in words_to_plot if word in word_to_id]
    
    # Get embeddings for these words
    embed_subset = embeddings[indices]
    
    # Reduce to 2D using t-SNE
    # Adjust perplexity based on number of samples
    n_samples = len(embed_subset)
    perplexity = min(30, n_samples - 1)  # t-SNE requires perplexity < n_samples
    
    if n_samples > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embed_2d = tsne.fit_transform(embed_subset)
        
        # Plot
        plt.figure(figsize=(12, 8))
        for i, word in enumerate(words_to_plot):
            if word in word_to_id:
                x, y = embed_2d[i]
                plt.scatter(x, y)
                plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points')
        
        plt.title("Word Embeddings Visualization")
        plt.show()
    else:
        print("Not enough samples for t-SNE visualization (need at least 2)")

# Test script
if __name__ == "__main__":
    # Load tokenizer from week 1
    from week1_tokenization.tokenizer import CustomTokenizer
    
    # Create a larger, more diverse corpus for meaningful embeddings
    corpus = [
        # Animals
        "the cat sat on the mat",
        "the dog played in the park",
        "birds fly in the sky",
        "fish swim in the water",
        "horses run in the field",
        "cows graze in the meadow",
        "sheep roam the hills",
        "pigs roll in mud",
        "ducks swim in ponds",
        "geese fly south",
        
        # Colors
        "red roses bloom in spring",
        "blue sky stretches above",
        "green grass grows tall",
        "yellow sun shines bright",
        "purple flowers smell sweet",
        "orange fruit tastes good",
        "pink clouds float by",
        "brown earth feels warm",
        "white snow falls gently",
        "black night covers all",
        
        # Emotions
        "happy people laugh loudly",
        "sad children cry softly",
        "angry men shout loudly",
        "excited girls jump high",
        "worried parents wait anxiously",
        "calm monks meditate quietly",
        "nervous students study hard",
        "proud parents smile warmly",
        "frustrated workers complain bitterly",
        "relaxed cats sleep peacefully",
        
        # Actions
        "quick runners sprint fast",
        "slow walkers stroll leisurely",
        "strong workers lift heavy",
        "weak children struggle hard",
        "brave soldiers fight fiercely",
        "careful drivers drive safely",
        "creative artists paint beautifully",
        "smart students learn quickly",
        "kind nurses help gently",
        "busy cooks prepare meals",
        
        # Objects
        "wooden tables stand sturdy",
        "metal cars drive fast",
        "plastic bottles hold water",
        "glass windows reflect light",
        "paper books contain knowledge",
        "stone walls protect homes",
        "cloth clothes keep warm",
        "leather shoes last long",
        "silver jewelry sparkles bright",
        "gold coins shine brightly",
        
        # Nature
        "tall trees grow slowly",
        "small flowers bloom quickly",
        "deep oceans hold secrets",
        "high mountains touch clouds",
        "wide rivers flow steadily",
        "narrow paths wind through",
        "thick forests hide animals",
        "thin ice breaks easily",
        "rough rocks feel hard",
        "smooth stones feel soft"
    ]
    
    # Tokenize and encode
    tokenizer = CustomTokenizer(min_freq=1)
    tokenizer.build_vocab(corpus)
    encoded_corpus = [tokenizer.encode(text) for text in corpus]
    
    # Create dataset
    dataset = SkipGramDataset(encoded_corpus, window_size=2)
    print(f"Training pairs: {len(dataset)}")
    
    # Create and train model
    vocab_size = len(tokenizer.word_to_id)
    embedding_dim = 50
    model = Word2Vec(vocab_size, embedding_dim)
    
    # Train
    train_word2vec(model, dataset, vocab_size, epochs=100)
    
    # Get embeddings
    embeddings = model.get_embeddings()
    
    # Find similar words
    def find_similar_words(word, embeddings, word_to_id, top_k=5):
        if word not in word_to_id:
            return []
        
        word_idx = word_to_id[word]
        word_embed = embeddings[word_idx]
        
        # Compute cosine similarity
        similarities = np.dot(embeddings, word_embed) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_embed)
        )
        
        # Get top k similar words
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        id_to_word = {v: k for k, v in word_to_id.items()}
        
        return [(id_to_word[idx], similarities[idx]) for idx in similar_indices]
    
    # Test similarity with more diverse words
    test_words = ["cat", "red", "happy", "run", "wooden"]
    for word in test_words:
        similar = find_similar_words(word, embeddings, tokenizer.word_to_id)
        print(f"\nMost similar to '{word}':")
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.3f}")
    
    # Visualize embeddings
    print("\nGenerating embedding visualization...")
    visualize_embeddings(embeddings, tokenizer.word_to_id) 