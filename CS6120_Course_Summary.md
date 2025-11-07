# CS6120: Natural Language Processing - Course Content Summary
## Fall 2025

---

## Course Overview

This document provides a comprehensive summary of the CS6120 Natural Language Processing course, organized by weekly topics and structured around pre-class, in-class, and post-class activities. The course follows a hands-on, project-based approach using PyTorch and modern NLP techniques.

**Primary Textbook:** Speech and Language Processing by Dan Jurafsky and James H. Martin
**Code Repository:** https://github.com/aanchan/cs6120-fall2025-public

---

## Week 1: Introduction to NLP

### Topics Covered
- Course mechanics and syllabus overview
- Introduction to Natural Language Processing
- Words and tokens (SLP Chapter 2)
- Tokenization fundamentals
- Next token prediction

### Pre-class Work
- Read SLP Chapter 2: [Words and Tokens](https://web.stanford.edu/~jurafsky/slp3/2.pdf)
- Review course syllabus and mechanics

### In-class Activity
- Set up development environment (git, SSH keys)
- Clone course repository: `git@github.com:aanchan/cs6120-fall2025-public.git`
- Run tokenizer code
- Explore regular expressions for tokenization: `re.findall(r'\b\w+\b|[.!?;,]')`
- Investigate alternative tokenization approaches
- Discuss tokenization for non-English languages

### Post-class Activity
**Levenshtein Distance (Edit Distance)**
- Calculate edit distance between strings (COURSE ↔ MODULE)
- Understand operations: insertion, deletion, substitution
- Analyze cases where diagonal values (i-1, j-1) are used
- Apply weighted edit distances with custom costs:
  - ADD: 1.0
  - REPLACE: 1.5
  - DELETE: 0.5
- Convert "DENVER" to "NERVED" with weighted costs

**Key Concepts:**
- Tokenization methods and regex patterns
- Edit distance algorithms
- String similarity metrics
- Dynamic programming for sequence alignment

---

## Week 2: Edit Distances and Tokenization

### Topics Covered
- ELIZA (early chatbot)
- Tokenization techniques (Dan Jurafsky's slides)
- Edit distances and minimum edit distance algorithm
- Word embeddings introduction
- PyTorch basics and autograd

### Pre-class Work
**Byte-Pair Encoding (BPE) Implementation**
- Study SLP textbook Section 2.4 on subword tokenization
- Implement BPE training algorithm (Figure 2.6)
- Use `tiny_shakespeare.txt` dataset
- Compare BPE implementation with textbook algorithm
- Apply BPE tokenization and compare with regex tokenizer
- Document differences between tokenization approaches

### In-class Activity
**Word Embeddings Exploration**
- Pull latest code from repository
- Set up virtual environment and install requirements
- Run `week2_embeddings/embeddings.py`
- Understand the code:
  - What does `np.dot()` do?
  - What does the `embeddings` variable contain?
  - What are the dimensions of embeddings?
  - How are high-dimensional embeddings visualized in 2D?
  - What characteristics appear in the embedding plots?

### Post-class Activity
**Comparing Tokenizers with Embeddings**
- Retrain embeddings using your BPE tokenizer (from pre-class)
- Train embeddings with both regex and BPE tokenizers on `tiny-shakespeare`
- Compare:
  - Number of embeddings from each tokenizer
  - Whether token counts match embedding counts
  - Generate and analyze 2D plots for both embedding sets
  - Identify similarities and differences in the plots

**Key Concepts:**
- Subword tokenization (BPE)
- Word2vec embeddings
- Vector representations of words
- Dimensionality reduction for visualization
- Tokenizer impact on embedding quality

---

## Week 3: Embeddings

### Topics Covered
- Word2vec embeddings (skip-gram and CBOW models)
- Cost functions for embedding training
- Neural network classifiers
- PyTorch for NLP tasks

**Course Materials:**
- Slides: [Vector Semantics](https://web.stanford.edu/~jurafsky/slp3/slides/vector25aug.pdf)
- Textbook: [SLP Chapter 5](https://web.stanford.edu/~jurafsky/slp3/5.pdf)

### Pre-class Work
**Skip-gram and CBOW Models**
- Study skip-gram model for word2vec embeddings
- Identify and write the cost function for skip-gram model
- Analyze Figures 5.6 and 5.7 from the textbook
- Read the original word2vec paper by Tomas Mikolov et al.: [Efficient Estimation of Word Representations](https://arxiv.org/pdf/1301.3781)
- Understand Continuous Bag of Words (CBOW) model:
  - Write the CBOW cost function
  - Explain differences from skip-gram
  - Draw network architecture
  - Explain model design choices
- Cite all sources used

### In-class Activity
**Round-table Discussion**
- Review Week 2 in-class activity (embedding code)
- Review Week 2 post-class activity (tokenizer comparison with embeddings)
- Review Week 3 pre-class activity (skip-gram and CBOW models)
- Discuss neural network classifiers
- Introduction to PyTorch for NLP
- Group discussions on embedding models

### Post-class Activity
- Continue working through PyTorch tutorials
- Practice implementing neural network classifiers
- Apply embeddings to classification tasks

**Key Concepts:**
- Skip-gram model architecture and loss function
- CBOW model architecture and loss function
- Differences between skip-gram and CBOW
- Training word embeddings
- Neural network classifiers with PyTorch

---

## Week 4: Neural Networks Foundations

### Topics Covered
- Deep learning with PyTorch
- Neural network fundamentals
- Building classification models

### Pre-class Work
- Review PyTorch Deep Learning tutorial from previous class session
- Reinforce understanding of PyTorch basics

### In-class Activity
- Hands-on PyTorch implementation exercises
- Building and training neural networks

### Post-class Activity
- Continue PyTorch exercises
- Implement neural network models

**Key Concepts:**
- PyTorch tensor operations
- Building neural network layers
- Training loops and optimization

---

## Week 5 & 6: Linear and Non-linear Classification

### Topics Covered
- Logistic regression
- Autograd with PyTorch
- Linear classifiers
- Neural networks with non-linear activation functions
- Optimization techniques
- Gradient descent and variants

**Course Materials:**
- [PyTorch Logistic Regression Tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html)
- [Autograd Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Logistic Regression Slides](https://web.stanford.edu/~jurafsky/slp3/slides/logreg25aug.pdf)
- [CS231n Linear Classification Notes](https://cs231n.github.io/linear-classify/)
- [Linear Classification Demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)
- [ConvNetJS Demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
- [Neural Networks Slides](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)

### Week 5 Pre-class Activity
**PyTorch Deep Learning Tutorial**
1. Complete PyTorch Deep Learning tutorial
2. Use Google Colab notebook
3. Submit Colab notebook link

**Supervised Text Classification Project**
1. Select a binary text classification dataset
2. Document dataset details and cite sources
3. Split data into training, validation, and test sets
4. Adapt PyTorch tutorial to your dataset
5. Create plots showing:
   - Training loss vs. epochs
   - Validation loss vs. epochs
6. Name the loss function used
7. Evaluate model performance on test set
8. Compare test vs. validation performance
9. Submit Colab notebook or GitHub repository link

### Week 6 Pre-class Activity
**Conceptual Deep Dive (Textbook Study)**

Study SLP Chapters 4 and 6 and answer the following:

1. **Softmax Probabilities (Section 4.7)**
   - How does softmax transform vectors to probability distributions?

2. **Cost Functions in Logistic Regression**
   - Identify different cost function formulations
   - Calculate cross-entropy loss example (ŷ=0.70, y=1)

3. **Gradient Descent**
   - Explain the mechanism
   - Write weight update equations
   - Define learning rate (η) as hyperparameter
   - Explain partial derivatives as vectors
   - Analyze Figure 4.4 (axes, curve characteristics)
   - Compare single-layer vs. multi-layer network curves

4. **Stochastic Gradient Descent (SGD)**
   - Define "stochastic" in this context
   - Differentiate from standard gradient descent
   - Explain mini-batch SGD and its importance

5. **Cross-validation**
   - Find 10-fold cross-validation figure
   - Explain use cases

6. **Chain Rule**
   - Locate in textbook
   - Explain utility in neural networks

7. **Non-linear Activation Functions**
   - Identify equations necessitating non-linearity

8. **Vectorization/Matrix Operations**
   - How to process m examples simultaneously
   - Avoiding for loops in implementation

9. **Embeddings as Neural Network Input**
   - How embedding vectors are selected
   - Role of matrix multiplication

### Week 6 Post-class Activity

**Part 1: TensorFlow Playground Experiments (7 points)**

Explore neural network behavior at [TensorFlow Playground](https://playground.tensorflow.org/):

1. Single neuron with linear activation on Circle dataset - describe limitations
2. Single neuron with Sigmoid activation on Circle dataset - assess classification ability
3. Multiple neurons in same layer - explore decision boundary complexity
4. Switch to Spiral dataset - test generalization
5. Modify network architecture - propose solutions for Spiral dataset
6. Implement ReLU activation - compare convergence with Tanh/Sigmoid
7. Analyze ReLU convergence speed

**Part 2: Classification Heads (3 points)**
- Watch instructional video on BERT as feature extractor
- Understand deep neural networks as automatic feature extractors
- Analyze how BERT converts to classification model
- Support with video screenshots

**Part 3: BERT and GPT Analysis (2 points)**
- Review CoNLL 2024 research paper
- Analyze Figure 1 relationship to course concepts
- Identify how GPT-BERT cost functions combine

**Part 4: Language Models (2 points)**
- Define language model function
- Determine if neural networks are necessary
- Support with textbook references

**Key Concepts:**
- Logistic regression for text classification
- Loss functions and cost functions
- Gradient descent and stochastic gradient descent
- Learning rate and hyperparameter tuning
- Activation functions (linear, sigmoid, tanh, ReLU)
- Backpropagation and chain rule
- Cross-validation techniques
- Neural network depth and width
- Feature extraction in deep learning
- BERT and GPT architectures

---

## Week 7: N-Grams and Language Models

### Topics Covered
- N-gram language models
- Large Language Models (LLMs)
- Optimizers and gradient descent review
- Language model evaluation (perplexity)

**Course Materials:**
- [N-Grams Slides](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf)
- [LLMs Slides](https://web.stanford.edu/~jurafsky/slp3/slides/llm25aug.pdf)
- [Google Books N-Gram Viewer](https://books.google.com/ngrams/info)

### Pre-class Activity
**Textbook Deep Dive (9 parts, 1 point each)**

Study SLP [Chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) and [Chapter 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf):

1. **Bi-gram Probabilities**
   - Define bi-grams
   - Explain probability estimation methods
   - Analyze Figure 3.2 (bi-gram probability table)
   - Identify mathematical assumptions for sentence probability calculation

2. **Language Model Evaluation**
   - Find section on evaluating language models
   - Identify the metric that starts with "P" (Perplexity)

3. **Perplexity Calculation**
   - Identify equation using unigram probabilities

4. **Figures 7.1 and 7.2**
   - Describe what these figures illustrate

5. **Autoregressive vs. Non-autoregressive Models**
   - Identify language model type in Figures 7.1 and 7.2 (starts with "A")
   - Explain the difference (e.g., BERT is non-autoregressive)

6. **Architecture Types**
   - Find section on encoders, decoders, and encoder-decoder architectures
   - Screenshot relevant portions for each architecture

7. **Sentiment Detection as Next Token Prediction**
   - Locate section on reframing sentiment detection
   - Identify the movie star mentioned in the example

8. **Zero-shot and Few-shot Prompting**
   - Find figures showing zero-shot and few-shot examples
   - Describe NLP tasks formulated as next token prediction

9. **Figure 7.13 Analysis**
   - Explain what this figure conveys

### In-class Activity
- Review Week 6 pre-class activity solutions
- Review Week 6 post-class activity solutions
- Deep dive into optimizers and gradient descent
- Discuss N-gram language models
- Explore LLM concepts

### Post-class Activity
- Continue exploring N-gram implementations
- Experiment with Google Books N-Gram Viewer
- Practice language model evaluation

**Key Concepts:**
- N-gram language models (unigram, bigram, trigram)
- Language model probability estimation
- Perplexity as evaluation metric
- Autoregressive vs. non-autoregressive models
- Encoder, decoder, and encoder-decoder architectures
- Zero-shot and few-shot learning
- Next token prediction paradigm
- Large Language Models fundamentals

---

## Week 8: Self-Attention and Residual Connections

### Topics Covered
- Self-attention mechanisms
- Residual connections
- Loss landscapes
- Transformer architecture introduction
- Static vs. dynamic embeddings

**Course Materials:**
- [Loss Landscapes with Residual Connections](https://www.cs.umd.edu/~tomg/projects/landscapes/)
- [Self-Attention by Hand](https://www.byhand.ai/p/11-can-you-calculate-self-attention)
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### Pre-class Activity
- Review materials on self-attention
- Prepare questions about attention mechanisms

### In-class Activity
**Static vs. Dynamic Embeddings**

1. **Code Exploration**
   - Clone/update repository: `https://github.com/aanchan/cs6120-fall2025-public`
   - Run in BERT mode: `python demo_static_vs_dynamic.py --mask bidirectional`
   - Run in GPT mode: `python demo_static_vs_dynamic.py --mask causal`

2. **Analysis Questions**
   - Compare static word2vec-style embeddings vs. dynamic contextual embeddings
   - Use terminal screenshots to illustrate differences
   - Identify code lines implementing self-attention
   - Connect implementation to self-attention calculation reference

3. **Key Observations**
   - How do static embeddings differ from contextual embeddings?
   - Why is positional encoding important?
   - What is the difference between bidirectional (BERT) and causal (GPT) masking?

### Post-class Activity
- Review Jay Alammar's Illustrated Transformer blog post
- Practice calculating self-attention by hand
- Explore how residual connections affect loss landscapes

**Key Concepts:**
- Self-attention mechanism
- Query, Key, Value matrices
- Attention scores and softmax
- Multi-head attention
- Residual connections and skip connections
- Loss landscape smoothing
- Static vs. dynamic (contextual) embeddings
- Positional encoding
- Bidirectional vs. causal attention masking
- BERT vs. GPT architectural differences

---

## Week 9: The Annotated Transformer Deep Dive

### Topics Covered
- Complete transformer architecture
- Attention is All You Need paper implementation
- Encoder-decoder structure
- Practical transformer implementation

**Course Materials:**
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Course Code Repository](https://github.com/aanchan/cs6120-fall2025-public/tree/main/week9_annotated_transformer)

### Pre-class Activity
**Annotated Transformer Study**

1. Work through the complete Annotated Transformer blog post
2. Run the code (available in course repository)
3. Post at least three questions about concepts you didn't understand
4. Share additional resources you found helpful:
   - Online articles, videos, or tutorials
   - Textbook sections (with screenshots)
   - Other explanatory materials

### In-class Activity
- Discuss pre-class questions as a group
- Deep dive into transformer architecture components:
  - Multi-head attention layers
  - Position-wise feed-forward networks
  - Layer normalization
  - Residual connections
  - Positional encoding
- Walk through code implementation
- Discuss encoder-decoder attention

### Post-class Activity
- Continue studying the Annotated Transformer
- Implement components of the transformer
- Experiment with the code

**Key Concepts:**
- Complete transformer architecture
- Encoder stack (6 layers)
- Decoder stack (6 layers)
- Multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward networks
- Layer normalization
- Positional encoding
- Masked self-attention in decoder
- Training and inference procedures
- Attention visualization



