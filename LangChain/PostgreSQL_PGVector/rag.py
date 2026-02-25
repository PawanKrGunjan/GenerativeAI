# modern_rag_chatbot.py
from dotenv import load_dotenv
import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# IMPORTANT: use langchain (not langchain_classic) for these
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# -----------------------------
# Config
# -----------------------------
DB_PASSWORD = os.getenv("POSTGRE_PASSWORD")
CONNECTION = f"postgresql+psycopg://postgres:{DB_PASSWORD}@localhost:5433/rag_test"
DOC_COLLECTION = "docs"
HISTORY_COLLECTION = "chats"
K_DOCS = 3
K_HISTORY = 5
SESSION_ID = "user_session_1"

# -----------------------------
# Embeddings & Vectorstores
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs_vectorstore = PGVector.from_existing_index(
    embedding=embeddings,
    collection_name=DOC_COLLECTION,
    connection=CONNECTION,
)
history_vectorstore = PGVector.from_existing_index(
    embedding=embeddings,
    collection_name=HISTORY_COLLECTION,
    connection=CONNECTION,
)

docs_retriever = docs_vectorstore.as_retriever(search_kwargs={"k": K_DOCS})
history_retriever = history_vectorstore.as_retriever(search_kwargs={"k": K_HISTORY})

# -----------------------------
# LLM (streaming)
# -----------------------------
llm = ChatOllama(
    model="llama3.2:3b",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# -----------------------------
# Prompt (expects: context, chat_history, input)
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the context and chat history to answer.\n\n"
            "Context:\n{context}\n\n"
            "Chat history:\n{chat_history}\n",
        ),
        ("human", "{input}"),
    ]
)

# -----------------------------
# Modern RAG chain
# -----------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(docs_retriever, document_chain)

# -----------------------------
# Terminal loop
# -----------------------------
print("RAG Chatbot (type 'exit' to quit):\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # 1) Retrieve history (semantic; see note below)
    history_docs = history_retriever.invoke(f"session:{SESSION_ID}")
    chat_history_text = "\n".join(doc.page_content for doc in history_docs)

    # 2) Stream answer (StreamingStdOutCallbackHandler prints tokens)
    _ = rag_chain.invoke(
        {
            "input": user_input,
            "chat_history": chat_history_text,
        }
    )
    print("\n")  # newline after streaming

    # 3) Persist chat (ensure session id is in text so it’s retrievable)
    history_vectorstore.add_documents(
        [
            Document(
                page_content=f"session:{SESSION_ID}\nHuman: {user_input}\n",
                metadata={"session": SESSION_ID, "role": "human"},
            ),
            # Optional: store the AI answer too; to do that you need the final answer text.
            # With streaming stdout callback, easiest is to switch to an accumulating callback.
        ]
    )

    print("---\n")
