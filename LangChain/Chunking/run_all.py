import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)


def main():
    # File Path (same as notebook: uses current working directory + filename)
    pdf_file_path = os.path.join(os.getcwd(), "pawankrgunjan.pdf")
    print("PDF path:", pdf_file_path)

    # Read the documents
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    print("Number of pages:", len(documents))

    # A. Text structure-based (RecursiveCharacterTextSplitter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print("Number of chunks:", len(chunks))

    # Notebook printed the last 5 chunks; keep the same idea here
    print("Last 5 chunks:")
    for i, chunk in enumerate(chunks[-5:]):
        print("-" * 50)
        print(f"Chunk {i+1}:\n{chunk}\n")

    # Splitting code text splitter integration guide
    print("Supported languages:", [e.value for e in Language])

    print(
        "Python separators:",
        RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON),
    )

    # Python example
    PYTHON_CODE = """
def hello_world():
 print("Hello, World!")

# Call the function
hello_world()
""".strip(
        "\n"
    )

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=50,
        chunk_overlap=0,
    )
    python_docs = python_splitter.create_documents([PYTHON_CODE])
    print("Python chunks:", python_docs)

    # JS example
    JS_CODE = """
function helloWorld() {
 console.log("Hello, World!");
}

// Call the function
helloWorld();
""".strip(
        "\n"
    )

    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=60,
        chunk_overlap=0,
    )
    js_docs = js_splitter.create_documents([JS_CODE])
    print("JS chunks:", js_docs)

    # NOTE: Your notebook continues with more cells (HTML, etc.).
    # If you want, paste the remaining part (or re-upload the notebook fully)
    # and I’ll extend this .py accordingly.


if __name__ == "__main__":
    main()
