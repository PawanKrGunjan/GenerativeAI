# LangChain Quick Reference Notes  

This practical cheat-sheet covers **LangChain**, the most widely adopted open-source framework for building applications powered by LLMs. It provides modular components for chains, agents, memory, retrieval (RAG), tools, callbacks, and orchestration — forming the foundation for many agentic systems.

LangChain is not a complete agent framework by itself but the **building blocks** used by LangGraph, LlamaIndex, CrewAI, and thousands of production apps. It excels at composability, debugging (via LangSmith), and integrating with 100+ providers and data sources.

**Key Highlights (2026 status):**
- Core + ecosystem: langchain, langchain-core, langchain-community, langgraph, langsmith
- 150+ integrations (LLMs, embeddings, vector stores, tools, document loaders)
- LCEL (LangChain Expression Language) for composable chains
- Production-ready: LangSmith (tracing, eval, datasets), LangGraph (stateful agents)
- Main Repo: https://github.com/langchain-ai/langchain
- Docs: https://python.langchain.com
- Install: `pip install langchain`

---

### 1. Basic Setup (Python)

```python
# Core + common integrations
pip install langchain langchain-openai langchain-community

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Environment
os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # Optional: LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_key"

# LLM & Embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Local models (Ollama example)
# pip install langchain-ollama
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# llm = ChatOllama(model="llama3.2")
```

---

### 2. Simple Chain (LCEL)

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"input": "Explain LangChain in 3 sentences."})
print(response)
```

**Streaming:**

```python
for chunk in chain.stream({"input": "Tell a joke"}):
    print(chunk, end="", flush=True)
```

---

### 3. Retrieval-Augmented Generation (RAG)

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# Load & split
loader = TextLoader("data/report.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Vector store
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# RAG chain
rag_prompt = ChatPromptTemplate.from_template(
    "Answer based only on this context:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What are the key findings?")
print(response)
```

---

### 4. Agents & Tools

```python
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

tools = [multiply]
agent = create_tool_calling_agent(llm, tools, prompt)  # Custom prompt needed
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "What is 7 times 8?"})
```

**Built-in tools:** Search (SerpAPI, Tavily), Wikipedia, Python REPL, SQL, etc.

---

### 5. Memory & Chat History

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Hi!")
history.add_ai_message("Hello! How can I help?")

chain_with_memory = (
    RunnablePassthrough.assign(messages=lambda x: history.messages)
    | prompt
    | llm
)
```

**Persistent memory:** Use Redis, SQL, etc. via `ConversationBufferMemory`.

---

### 6. Advanced Features

- **LangGraph** → Stateful multi-agent graphs (see separate notes)
- **LangSmith** → Tracing, evaluation, datasets, prompt playground
- **Callbacks** → Streaming tokens, cost tracking
- **Evaluation** → Automated QA, relevance scoring
- **Document Loaders** → 100+ sources (PDF, web, YouTube, GitHub, etc.)
- **Output Parsers** → JSON, Pydantic, structured responses

---

### 7. Best Practices Summary

| Goal                                | Recommended Approach                                      |
|-------------------------------------|-----------------------------------------------------------|
| Simple prompt → response            | LCEL chain (prompt | llm | parser)                        |
| RAG over private docs               | Document loader → splitter → vectorstore → retriever chain |
| Tool-using agent                    | create_tool_calling_agent + custom tools                  |
| Chat with memory                    | ChatMessageHistory or buffer memory                       |
| Production monitoring               | LangSmith tracing + evaluation                            |
| Complex stateful agents             | Use LangGraph (built on LangChain)                        |

---

### 8. Recommended Models & Integrations (2026)

| Category   | Example                           | Use Case                          |
|------------|-----------------------------------|-----------------------------------|
| LLM        | gpt-4o-mini, claude-3.5-sonnet    | Best quality/cost                 |
| Local      | llama3.2, granite-3.3 (via Ollama)| Privacy/local                     |
| Embedding  | text-embedding-3-small            | Default OpenAI                    |
| Vector DB  | Chroma, Pinecone, PGVector        | Fast retrieval                    |
| Search     | Tavily, Serper                    | Web search tools                  |

---

**You now have a solid LangChain foundation!**  
LangChain is the **Swiss Army knife** of LLM application development — use it directly for chains/RAG or as the backbone for higher-level frameworks like LangGraph.

Explore templates: https://github.com/langchain-ai/langchain/tree/master/templates  
Docs & API: https://python.langchain.com/docs/

Happy chaining! 🔗  

**Prepared By: Pawan Kumar Gunjan**  
**Date: January 09, 2026**