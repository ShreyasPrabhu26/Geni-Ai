import tiktoken

model_name = 'gpt-4o'

encoder = tiktoken.encoding_for_model(model_name)

print(f"Vocab Size of {model_name} is ",encoder.n_vocab)

text_input="Hello Shreyas"
tokens = encoder.encode(text_input)

print("Tokens",tokens)

# ---------------------------

sample_tokens = [13225, 1955, 14603, 288]
decoded= encoder.decode(sample_tokens)
print("Decoded",decoded)