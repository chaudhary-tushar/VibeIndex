The primary role of parser.py is to act as a "Code Parser & Chunker" as defined in
  your Mermaid diagram. It should read various file types, parse them, and break them
  down into smaller, meaningful chunks that can be used by the downstream embedding and
  retrieval layers.

  Here are the recommended implementation steps for src/preprocessing/parser.py:

  1. Implement a Central Parsing Dispatcher

  Create a main function (e.g., parse_file(file_path: str)) that acts as a dispatcher.
  This function should:
   * Determine the file type based on its extension or MIME type (e.g., .py, .md, .pdf,
     .csv).
   * Call a specific parsing function for that file type (e.g., _parse_python(),
     _parse_markdown()).
   * Return a list of structured "chunk" objects. Each chunk could contain the content,
     file path, and any extracted metadata (like function name or section heading).

  This approach is inspired by the snippet def _get_data_file_preprocessing_code(file:
  File), which checks file.mime_type to decide how to process a file.

  2. Add Markdown Processing Logic

  As described in the snippet [Title] : Heading Processes Markdown content..., you need
  to handle documentation files. Your parser.py should include a function to:
   * Read the content of a Markdown file.
   * Parse the Markdown structure. You can use a library like markdown-it-py.
   * Chunk the content based on headings, sections, or paragraphs. This will create more
     focused and contextually relevant chunks for embedding.

  3. Add PDF Processing Logic

  Similar to Markdown, the snippet [Title] : Heading Processes PDF documents... indicates
  the need to process PDFs. You should implement a function that:
   * Uses a library like PyMuPDF (fitz) to open and read PDF files.
   * Extracts text content from each page.
   * Chunks the extracted text. You can create one chunk per page or try to split the text
     into paragraphs.

  4. Implement Code Parsing and Chunking

  This is a core requirement from your Mermaid diagram. For source code files (e.g.,
  Python), the parser should:
   * Read the source code file.
   * Use an Abstract Syntax Tree (AST) parser (like Python's built-in ast module) to
     analyze the code structure.
   * Extract meaningful code blocks such as functions, classes, and methods. Each block
     should become a separate chunk.
   * This will allow the RAG pipeline to find and retrieve specific functions or classes
     when answering a query.

  5. Handle Data Files

  The snippet def _get_data_file_preprocessing_code(file: File) suggests a strategy for
  handling data files like CSVs. Instead of just reading the raw content, you can:
   * Implement a function that, for a given data file, generates a small script to load
     and analyze it (e.g., using pandas to get column names, data types, and a sample of
     rows).
   * The output of this analysis script can then be treated as the "parsed" content for
     the data file, providing a useful summary for the RAG pipeline.
