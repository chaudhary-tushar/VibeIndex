from src.embedding.embedder import embed_code
from src.preprocessing.parser import parse_code


def main():
    # TODO: Implement logic to read files from the code repository
    code_files = []

    for file in code_files:
        # 1. Parse the code
        parsed_code = parse_code(file)

        # 2. Embed the code
        embeddings = embed_code(parsed_code)

        # 3. Store the embeddings in Qdrant
        # TODO: Implement Qdrant storage logic


if __name__ == "__main__":
    main()
