# mintoken
Personal Projects 

# Minimal (byte-level) Byte Pair Encoding tokenizer.

**Reference: GPT-2 tokenizer and Andrej Karpathy's minbpe.**


Previously I used **character-level** tokenization in nanoGPT. Now we should try to make a clean code for the **(byte-level)** Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This algorithm was popularized for LLMs by the GPT-2 paper and the associated GPT-2 code release from OpenAI. Sennrich et al. 2015 is cited as the original reference for the use of BPE in NLP applications. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use this algorithm to train their tokenizers.
