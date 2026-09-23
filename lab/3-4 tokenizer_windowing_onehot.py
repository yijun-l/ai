from pytorch.CharTokenizer import CharTokenizer
from pytorch.OneHot import OneHot
from pytorch.WindowDataset import WindowDataset
from pytorch.Embedding import Embedding

if __name__ == "__main__":
    # Note: If 'tokenizer.json' does not exist, run '3-3 char_level_tokenizer.py' first to build the vocabulary.

    tokenizer = CharTokenizer()
    sample_text = "To be, or not to be, that is the question."
    ids = tokenizer.encode(sample_text)

    window_dataset = WindowDataset(context_length=5)
    x_ids, y_ids = window_dataset.build(ids)

    one_hot = OneHot(vocab_size=tokenizer.vocab_size)
    x_one_hot = one_hot.forward(x_ids)

    # Output verification
    print(f"Token IDs length: {len(ids)}")
    print(f"X shape: {x_ids.shape}, Y shape: {y_ids.shape}")
    print(f"One-hot tensor shape (B, T, V): {x_one_hot.shape}")

    # Embedding
    embed_dim = 16
    embedding = Embedding(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        seed=0,
    )
    x_emb = embedding.forward(x_one_hot)

    print(f"Embedding tensor shape (B, T, D): {x_emb.shape}")
    print(f"Embedding weight shape (V, D): {embedding.weight.shape}")