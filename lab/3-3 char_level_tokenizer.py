from pytorch.CharTokenizer import CharTokenizer

# Corpus covering all uppercase, lowercase, digits, and ASCII symbols
FULL_ENGLISH_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "  # All 26 lowercase letters
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS. "     # All 26 uppercase letters
    "0123456789 "                                    # All 10 digits
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n"         # Common ASCII symbols and whitespace
)

if __name__ == "__main__":
    # 1. Reset state and file before testing
    CharTokenizer.reset_all()
    tokenizer = CharTokenizer()

    # 2. Build vocabulary from corpus
    tokenizer.add_corpus(FULL_ENGLISH_CORPUS)

    # 3. Test encoding
    sample_text = "Talk is cheap. Show me the code."
    ids = tokenizer.encode(sample_text)
    print(f"\nOriginal Text: {sample_text}")
    print(f"Encoded IDs: {ids}\n")

    # 4. Test decoding
    decoded_text = tokenizer.decode(ids)
    print(f"Decoded Text: {decoded_text}")