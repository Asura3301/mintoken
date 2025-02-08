"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Reference: GPT-2 tokenizer and Andrej Karpathy's minGPT.
"""

from utils import Tokenizer, get_stats, merge
# import regex as re

# Regex
# gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class MinBPE(Tokenizer):
    
  def __init__(self):
    super().__init__()

  def train(self, text, vocab_size, verbose=False):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # convert input text to bytes
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0-255
        
        # create vocab
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
        # save class variables (took from minbpe basic.py)
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()


  def encode(self, text):
    """
    given a string,
    return list of int(the tokens)
    """
    tokens = list(self.text.encode("utf-8"))
    while len(tokens) >= 2:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # in python when you call min to iterator, we iterating only keys of this dict
      if pair not in self.merges:
        break # nothing else can be merged
      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx)

    return tokens


  def decode(self, ids):
    """
    given ids: list of integers
    return: Python string
    """

    tokens = b"".join(self.vocab[idx] for idx in self.ids)
    text = tokens.decode("utf-8", errors="replace")
    return text