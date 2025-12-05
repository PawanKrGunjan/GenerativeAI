# rag_url_chatbot.py  ← Copy-paste and run (FIXED)

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# === Models ===
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# llm = ChatOpenAI(
#     base_url="https://api.perplexity.ai",
#     api_key=os.getenv("PERPLEXITY_API_KEY"),
#     model="sonar-pro",           # correct & powerful model
#     temperature=0.7
# )

llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",  # "HuggingFaceH4/zephyr-7b-beta"
    temperature=0.7,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=llm_endpoint)


print("🤗 HuggingFace Model loaded:")
print(f"   • Name : {llm_endpoint.repo_id}")

# Try to fetch the real number of parameters from the HF model card (fast & silent)
try:
    import json
    info_url = f"https://huggingface.co/{llm_endpoint.repo_id}/raw/main/config.json"
    resp = requests.get(info_url, timeout=8)
    if resp.status_code == 200:
        config = resp.json()
        # Most models store it as "num_parameters", "n_params" or under model_type specifics
        if "num_parameters" in config:
            params = config["num_parameters"]
        elif "n_params" in config:
            params = config["n_params"]
        elif hasattr(llm_endpoint.client, "model_config"):
            # fallback for some TGI servers
            params = llm_endpoint.client.model_config.get("model_type", {}).get("num_parameters")
        else:
            # Very common keys for popular models
            possible_keys = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
            if all(k in config for k in possible_keys):
                # Rough estimate for transformer models
                h = config["hidden_size"]
                l = config["num_hidden_layers"]
                a = config["num_attention_heads"]
                params = 12 * l * h * h + l * 4 * h * h  # approx
            else:
                params = None

        if params is not None:
            if params > 1e9:
                print(f"   • Parameters : {params / 1e9:.1f}B")
            elif params > 1e6:
                print(f"   • Parameters : {params / 1e6:.1f}M")
            else:
                print(f"   • Parameters : {params}")
        else:
            print("   • Parameters : (not listed in config.json)")
    else:
        print("   • Parameters : (could not fetch config.json)")
except Exception:
    print("   • Parameters : (unknown – quick lookup failed)")
print()  # empty line for readability

# === Q&A Prompt ===
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the following context. If you don't know, say: I don't know.\n\nContext:\n{context}"),
    ("human", "{input}")
])

# === Global variables ===
retriever = None
rag_chain = None

def load_url(url: str):
    global retriever, rag_chain

    print(f"\nFetching {url} ...")
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": url, "chunk": i}) for i, chunk in enumerate(chunks)]

    print(f"Split into {len(docs)} chunks → building vector database...")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_chain_local = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    rag_chain = rag_chain_local
    print("Vector database ready!")

    # === DEBUG: Test retriever first ===
    print("\n Testing retriever...")
    test_docs = retriever.invoke("What is this page about?")
    print(f"Retrieved {len(test_docs)} docs, first 200 chars: {test_docs[0].page_content[:200]}...\n")

    # === DEBUG: Test LLM standalone ===
    print(" Testing LLM...")
    try:
        test_response = llm.invoke([("human", "Say hello")])
        print(f"LLM works: {test_response.content[:100]}...\n")
    except Exception as llm_err:
        print(f"LLM failed: {llm_err}")
        return

    # === AUTO SUMMARY ===
    print("🤗 Generating title and summary...")
    summary_prompt = (
        "TITLE: [1 line title]\n"
        "SUMMARY:\n"
        "• [line 1]\n• [line 2]\n• [line 3]\n• [line 4]"
    )
    
    try:
        summary = rag_chain.invoke(summary_prompt)
        print("\n" + "="*60)
        print("🤗 SUCCESS:", summary)
        print("="*60 + "\n")
    except Exception as e:
        print(f"Full error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("🤗 You can now ask questions about this page!\n")



# === Chat Loop ===
print(" 🤗 Hugging Face RAG Chatbot Ready!")
print("Paste a URL → Auto summary → Ask questions!\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye! 👋")
        break

    if user_input.startswith(("http://", "https://")):
        try:
            load_url(user_input)
        except Exception as e:
            print(f"Error: {e}")
        continue

    if rag_chain is None:
        print("Please paste a URL first!")
        continue

    print("🧠 Thinking...")
    try:
        answer = rag_chain.invoke(user_input)
        print(f"💡 Answer: {answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")
