import json
from pathlib import Path


class CharTokenizer:
    """Singleton Character-Level Tokenizer with JSON persistence."""

    tokenizer_file = (
        Path(__file__).resolve().parent.parent / "tokenizer.json"
    )
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Enforce Singleton pattern to ensure a single shared instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize special tokens, mappings, and load/create vocabulary."""
        if self._initialized:
            return

        # Special Tokens configuration
        self.eos_token = "<EOS>"
        self.eos_id = 0

        self.unk_token = "<UNK>"
        self.unk_id = 1

        self.vocab = []
        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}

        # Load existing vocab or initialize empty
        if self.tokenizer_file.exists():
            self._load()
        else:
            self._init_empty_vocab()

        self._initialized = True

    @classmethod
    def reset_all(cls):
        """Delete saved JSON file and reset the singleton instance (useful for testing)."""
        if cls.tokenizer_file.exists():
            cls.tokenizer_file.unlink()

        if cls._instance is not None:
            cls._instance._initialized = False
            cls._instance = None

    def add_corpus(self, text: str):
        """Extract unique characters from text and append new ones to vocabulary."""
        text = text.strip()
        if not text:
            raise ValueError("text is empty")

        changed = False
        # Deduplicate while preserving character order
        for char in dict.fromkeys(text):
            if char not in self.token_to_id:
                self._add_token(char)
                changed = True

        if changed:
            self._save()

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        """Convert text into token IDs. Fallback to unk_id for unknown characters."""
        ids = []
        for char in text:
            token_id = self.token_to_id.get(char, self.unk_id)
            ids.append(token_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text. Stop decoding when eos_id is encountered."""
        tokens = []
        for token_id in ids:
            token_id = int(token_id)

            if token_id == self.eos_id:
                break

            tokens.append(self.id_to_token.get(token_id, self.unk_token))

        return "".join(tokens)

    def _init_empty_vocab(self):
        """Initialize vocabulary with fixed special tokens (<EOS>, <UNK>)."""
        self.vocab = [self.eos_token, self.unk_token]
        self.vocab_size = 2
        self.token_to_id = {
            self.eos_token: self.eos_id,
            self.unk_token: self.unk_id,
        }
        self.id_to_token = {
            self.eos_id: self.eos_token,
            self.unk_id: self.unk_token,
        }

    def _add_token(self, token: str) -> int:
        """Append a single character token to the vocabulary."""
        if token in self.token_to_id:
            return self.token_to_id[token]

        new_id = self.vocab_size
        self.vocab.append(token)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        self.vocab_size += 1
        return new_id

    def _save(self):
        """Save current vocabulary and mappings to JSON file."""
        data = {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "vocab": self.vocab,
        }

        with open(self.tokenizer_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Tokenizer saved. Current vocab size: {self.vocab_size}")

    def _load(self):
        """Load vocabulary and mappings from JSON file."""
        with open(self.tokenizer_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.vocab_size = len(self.vocab)

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        self.eos_token = data["eos_token"]
        self.unk_token = data["unk_token"]

        self.eos_id = self.vocab.index(self.eos_token)
        self.unk_id = self.vocab.index(self.unk_token)