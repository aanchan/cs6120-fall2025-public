import re
from collections import Counter
from typing import List, Dict, Tuple

class CustomTokenizer:
    """A simple regex-based tokenizer with vocabulary building."""
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.word_freq = Counter()
        
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization using regex."""
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b|[.!?;,]', text)
        
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Add words that meet minimum frequency
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        print(f"Vocabulary size: {len(self.word_to_id)}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = self.tokenize(text)
        return [self.word_to_id.get(token, self.word_to_id["<UNK>"]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_word.get(id, "<UNK>") for id in ids]
        return " ".join(tokens)

# Test script
if __name__ == "__main__":
    # Sample data
    texts = [
        "Hello world! How are you?",
        "I'm doing great, thanks!",
        "Natural language processing is fun.",
        "We're building a tokenizer."
    ]
    
    # Initialize and test tokenizer
    tokenizer = CustomTokenizer(min_freq=1)
    
    # Test basic tokenization
    print("Tokenization examples:")
    for text in texts[:2]:
        tokens = tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}\n")
    
    # Build vocabulary
    tokenizer.build_vocab(texts)
    
    # Test encoding/decoding
    test_text = "Hello, how are you doing?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}") 