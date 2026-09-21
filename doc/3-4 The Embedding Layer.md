# The Embedding Layer

After a **tokenizer** maps raw text into discrete sequences of numbers (Token IDs), the neural network faces a key challenge: **machine learning models work with continuous vector spaces, not discrete integers**.

The **Embedding Layer** serves as the bridge, turning these discrete tokens into the continuous vectors that neural networks need.

To keep matrix operations and tensor transformations clear, we define the following standard dimensions:

- **$B$ (Batch Size)**: The number of independent text sequences processed simultaneously in a single forward pass.
- **$T$ (Sequence Length / Time Steps)**: The number of tokens in a single input sequence.
- **$V$ (Vocabulary Size)**: The total count of unique discrete tokens supported by the model (e.g., V=32,000 for a typical BPE-based setup).
- **$D$ ($d_{\text{model}}$ / Embedding Dimension)**:  The size of the continuous vector representation for each token, defining the hidden feature space of the Transformer.

## Discrete Representation: One-Hot Encoding

Imagine we have a vocabulary of size $V$. In this state, a discrete Token ID like $97$ is just a label which has no geometric properties or mathematical meaning.

To use linear algebra, we must first turn this single number into a high-dimensional One-Hot Vector $x \in \mathbb{R}^{V}$:

$$x = [\underbrace{0, 0, \dots, 1}_{\text{index } i}, \dots, 0]$$

However, this simple mapping introduces two fundamental limitations:

- **High Sparsity and Dimensionality**: The vector's length grows directly with $V$. For large vocabularies, storing and processing these highly sparse vectors wastes a lot of memory and computing power.

- **Orthogonality and Uniform Distance**: By definition, any pair of different one-hot vectors is orthogonal:

    $$x_i \cdot x_j = 0 \quad (\forall i \neq j)$$
    
    This forces the model to treat every token as an isolated island, completely ignoring natural, intrinsic relationships (such as semantic similarities between words like "cat" and "dog").

## Mapping to Continuous Space

To overcome the flaws of One-Hot vectors, the model projects discrete symbols into a lower-dimensional, continuous feature space. It does this using an Embedding Matrix:

$$W_{\text{emb}} \in \mathbb{R}^{V \times D}$$

where $D$ represents the hidden embedding dimension ($D \ll V$).

`Discrete Token ID >>> [ One-Hot Vector (1 x V) ] × [ Embedding Matrix (V x D) ] >>> Embedding Vector (1 x D)`

This transformation maps a sparse One-Hot vector $x$ into a dense, continuous vector $e \in \mathbb{R}^{D}$:

$$e = x \cdot W_{\text{emb}}$$

Unlike fixed One-Hot representations, the weights of $W_{\text{emb}}$ are trainable parameters. Through backpropagation, the model learns these continuous representations, allowing semantically or functionally related tokens to naturally align closer together within the D-dimensional space.

### Matrix Multiplication vs. Table Lookup

While the formula $e = x \cdot W_{\text{emb}}$ is mathematically elegant and keeps the neural network differentiable, executing it this way in hardware would be an engineering disaster.

For a vocabulary of size 32,000, creating massive One-Hot vectors filled with 99.9% zeros just to perform a matrix multiplication wastes enormous memory and compute.

Instead, modern deep learning frameworks (like PyTorch's `nn.Embedding`)  treat the Embedding Matrix purely as a Lookup Table:

- **Forward Pass**: The model uses the discrete Token ID directly as an integer index to retrieve the corresponding row from the matrix. This completely bypasses matrix multiplication, achieving O(1) time complexity and saving billions of wasted calculations on GPUs.

- **Backward Pass**: During backpropagation, the engine uses gradient routing which directly passes gradients back to the specific rows that were retrieved, updating the matrix without ever building a sparse One-Hot vector.

## Injecting Order via Positional Encoding

While Token Embeddings capture the semantic meaning of each token, they carry **zero information about their position or order in a sentence**.

Because **Self-Attention** processes all tokens in parallel simultaneously, it is **permutation invariant**. Without explicit order signals, the model perceives the sentence `["the", "dog", "bit", "me"]` and its scrambled version `["me", "bit", "the", "dog"]` as identical sets of vectors.

To restore word order, a **Positional Embedding** $P \in \mathbb{R}^{T \times D}$ is injected by element-wise addition before the vectors are fed into the Transformer:

$$X_{\text{final}} = E_{\text{token}} + P_{\text{pos}}$$

$$\text{Shape: } (B, T, D) + (1, T, D) \longrightarrow (B, T, D)$$

### Implementation Strategies

To generate this position matrix $W_{\text{pos}}$, researchers commonly use one of two classic strategies:

#### 1. Learned Positional Embeddings (e.g., GPT-2/3)

Instead of using a formula, the model treats positions like a second vocabulary and learns them from scratch.

- A trainable matrix $W_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times D}$ is created, where each position index (from $0$ to $T_{\text{max}}-1$) gets its own static D-dimensional vector.
- It is simple to implement using `nn.Embedding(max_len, d_model)` in PyTorch, but the model cannot process sequences longer than the pre-defined limit ($T_{\text{max}}$).

#### 2. Sinusoidal Positional Encodings (Original Transformer)

Instead of learning parameters, the model calculates fixed wave-like patterns using sine and cosine functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$

 Because wave functions are infinite and continuous, this approach allows the model to extrapolate, meaning it can handle sentences longer than any it saw during training.