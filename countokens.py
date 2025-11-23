from transformers import AutoTokenizer

# 1. Define the model name
# Replace 'bert-base-uncased' with your specific model (e.g., 'gpt2', 'llama-2-7b', etc.)
model_name = "bert-base-uncased"
input_string = "Hello, I am using Hugging Face's tokenizer to count tokens."

# 2. Load the tokenizer for the specific model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenize the input string
# Setting return_tensors='pt' is optional, but common practice.
tokenized_output = tokenizer(input_string)

# 4. Calculate the token count
# The 'input_ids' tensor holds the numerical representation of the tokens.
# The shape[1] gives the length of the sequence (number of tokens).
token_count = tokenized_output["input_ids"]

# Display results
print(f"Model: {model_name}")
print(f"Input String: '{input_string}'")
print("-" * 30)
# This will show the actual tokens (e.g., ['hello', ',', 'i', 'am', ...])
print(f"Tokens (List): {tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0])}")
print(f"Token Count: {len(token_count)}")
