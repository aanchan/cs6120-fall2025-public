# NLP Course: Building Mini-LLM from Scratch

This repository contains a complete semester-long project where students build a language model from scratch, progressing from basic tokenization to a working transformer-based model.

## Project Structure

```
mini_llm_project/
├── week1_tokenization/
│   └── tokenizer.py          # Custom tokenizer implementation
├── week2_embeddings/
│   └── embeddings.py         # Word2Vec embeddings
├── week3_classification/
│   └── classifier.py         # Text classification model
├── week5_neural_lm/
│   └── neural_lm.py         # Neural language model
├── week6_attention/
│   └── attention.py         # Self-attention mechanism
├── week7_ngram/
│   └── ngram_lm.py         # Traditional n-gram model
├── week8_transformer/
│   └── transformer.py       # Full transformer implementation
├── week9_generation/
│   └── advanced_generation.py # Advanced generation techniques
├── final_project/
│   └── app.py              # Complete application
├── data/
│   └── tiny_shakespeare.txt # Sample training data
├── utils/
│   └── helpers.py          # Utility functions
└── README.md
```

## Learning Progression

### Week 1: Tokenization and Text Processing
- **Learning Objectives**: Understand different tokenization strategies, implement regex-based tokenization, handle special cases
- **Key Concepts**: Vocabulary building, encoding/decoding, special tokens
- **Implementation**: `CustomTokenizer` class with regex-based tokenization

### Week 2: Word Embeddings
- **Learning Objectives**: Understand word embeddings conceptually, implement Word2Vec skip-gram model
- **Key Concepts**: Skip-gram, negative sampling, cosine similarity
- **Implementation**: `Word2Vec` model with visualization

### Week 3: Text Classification
- **Learning Objectives**: Build intuition for classification, implement neural classifier
- **Key Concepts**: Mean pooling, neural architecture, sentiment analysis
- **Implementation**: `SimpleClassifier` with embedding layers

### Week 5: Neural Language Models
- **Learning Objectives**: Transition from classification to generation, build neural n-gram model
- **Key Concepts**: Teacher forcing, perplexity, temperature sampling
- **Implementation**: `NeuralLanguageModel` with context windows

### Week 6: Attention Mechanism
- **Learning Objectives**: Understand self-attention, implement scaled dot-product attention
- **Key Concepts**: Multi-head attention, positional encoding, attention visualization
- **Implementation**: `SelfAttention` and `AttentionLanguageModel`

### Week 7: N-gram Language Models (Baseline)
- **Learning Objectives**: Implement traditional n-gram model, understand smoothing
- **Key Concepts**: Laplace smoothing, Kneser-Ney, perplexity calculation
- **Implementation**: `NgramLanguageModel` with multiple smoothing options

### Week 8: Building a Transformer
- **Learning Objectives**: Combine all components, implement full transformer
- **Key Concepts**: Layer normalization, residual connections, causal masking
- **Implementation**: `MiniGPT` with complete transformer architecture

### Week 9: Advanced Generation and Applications
- **Learning Objectives**: Implement advanced sampling, fine-tune for tasks
- **Key Concepts**: Beam search, contrastive search, instruction following
- **Implementation**: `BeamSearchDecoder`, `ContrastiveDecoder`, `TextGenerator`

## Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib scikit-learn seaborn tqdm streamlit
```

### Quick Start

1. **Test Tokenization**:
```bash
cd week1_tokenization
python tokenizer.py
```

2. **Train Word Embeddings**:
```bash
cd week2_embeddings
python embeddings.py
```

3. **Run Text Classification**:
```bash
cd week3_classification
python classifier.py
```

4. **Train Neural Language Model**:
```bash
cd week5_neural_lm
python neural_lm.py
```

5. **Test Attention**:
```bash
cd week6_attention
python attention.py
```

6. **Compare with N-gram Baseline**:
```bash
cd week7_ngram
python ngram_lm.py
```

7. **Train Full Transformer**:
```bash
cd week8_transformer
python transformer.py
```

8. **Test Advanced Generation**:
```bash
cd week9_generation
python advanced_generation.py
```

## Key Features

### Tokenization (`week1_tokenization/tokenizer.py`)
- Regex-based tokenization with contraction handling
- Vocabulary building with minimum frequency threshold
- Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`

### Word Embeddings (`week2_embeddings/embeddings.py`)
- Skip-gram Word2Vec implementation
- Negative sampling for training
- t-SNE visualization of embeddings
- Similarity calculations

### Text Classification (`week3_classification/classifier.py`)
- Neural classifier with embedding layers
- Mean pooling for sentence representation
- Support for pre-trained embeddings
- Training and validation loops

### Neural Language Model (`week5_neural_lm/neural_lm.py`)
- Context-based language modeling
- Teacher forcing during training
- Temperature-controlled generation
- Perplexity evaluation

### Attention Mechanism (`week6_attention/attention.py`)
- Scaled dot-product attention
- Multi-head attention implementation
- Positional encoding
- Attention weight visualization

### N-gram Model (`week7_ngram/ngram_lm.py`)
- Traditional n-gram language model
- Multiple smoothing techniques (Laplace, Kneser-Ney)
- Perplexity calculation
- Text generation with temperature

### Transformer (`week8_transformer/transformer.py`)
- Complete GPT-style transformer
- Multi-head attention blocks
- Position embeddings
- Causal masking for autoregressive generation
- Advanced sampling (top-k, top-p)

### Advanced Generation (`week9_generation/advanced_generation.py`)
- Beam search decoding
- Contrastive search for diversity
- Instruction fine-tuning
- Interactive demo interface

## Model Architecture

### MiniGPT (Week 8)
```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, 
                 num_layers=6, ff_dim=1024, max_seq_len=512):
        # Token and position embeddings
        # Transformer blocks with attention
        # Output projection
```

### Key Components
- **Embeddings**: Token + positional embeddings
- **Transformer Blocks**: Multi-head attention + feed-forward
- **Generation**: Autoregressive with various sampling strategies

## Training Process

### Data Preparation
1. Load and tokenize text data
2. Build vocabulary from training corpus
3. Create training/validation splits
4. Prepare data loaders

### Training Loop
1. Forward pass through model
2. Calculate loss (cross-entropy)
3. Backward pass and gradient clipping
4. Optimizer step
5. Validation and perplexity calculation

### Evaluation Metrics
- **Perplexity**: Primary metric for language models
- **BLEU Score**: For generation quality
- **Diversity Score**: For generation variety

## Advanced Features

### Generation Strategies
- **Greedy**: Always select highest probability token
- **Sampling**: Sample from probability distribution
- **Beam Search**: Maintain multiple candidate sequences
- **Contrastive Search**: Balance likelihood and diversity

### Fine-tuning
- **Instruction Following**: Fine-tune for specific tasks
- **Domain Adaptation**: Adapt to specific text domains
- **Parameter-Efficient**: Use adapter layers


## Resources

- [Karpathy's makemore](https://github.com/karpathy/makemore)
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch tutorials](https://pytorch.org/tutorials/)

## Contributing

This is an educational project. Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features or examples
- Improve documentation

## License

This project is for educational purposes. Please respect the original sources and citations.

## Acknowledgments

- Based on modern NLP practices and transformer architecture
- Inspired by educational resources from Stanford, Harvard, and others
- Built with PyTorch and standard NLP libraries 