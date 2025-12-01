import argparse

from transformers import AutoTokenizer


def count_tokens(prompt: str, model_name: str) -> int:
    """
    Counts the number of tokens in a prompt using a specified Hugging Face model's tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(prompt)
        return len(tokens)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return 0


def main():
    """
    Main function to parse arguments and count tokens for various models.
    """
    parser = argparse.ArgumentParser(description="Count tokens for a given prompt across different models.")
    parser.add_argument("prompt", type=str, help="The prompt to count tokens for.")
    args = parser.parse_args()

    models = [
        "google/embeddinggemma-300m",
        "voyageai/voyage-code-2",
        "Qwen/Qwen3-Embedding-8B",
        "meta-llama/Llama-3.2-3B",
    ]

    print(f'Counting tokens for prompt: "{args.prompt}"')
    print("-" * 30)

    for model in models:
        token_count = count_tokens(args.prompt, model)
        if token_count > 0:
            print(f"{model}: {token_count} tokens")


if __name__ == "__main__":
    main()
