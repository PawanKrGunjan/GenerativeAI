import os
import logging
from typing import Literal, List, Dict, Any, Optional
from ddgs import DDGS
import trafilatura
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.DEBUG)

# Load .env first
load_dotenv()

# Option A: Use LangChain tracing env vars (works with LangSmith)
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    # If you stored the key as LANGSMITH_API_KEY in .env, map it for LangChain
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "deepagents-debug")

from typing import Literal, List, Dict, Any, Optional
from ddgs import DDGS
import trafilatura

def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    region: str = "in-en",
    timelimit: Optional[str] = "d",
) -> List[Dict[str, Any]]:
    """Search the web with DuckDuckGo and optionally extract page text.

    Args:
        query: Search query text.
        max_results: Number of results to return.
        topic: "general" uses text search, "news" uses news search.
        include_raw_content: If True, fetch each result URL and extract main text.
        region: DuckDuckGo region code (e.g., "in-en").
        timelimit: Time filter for DDG results: "d", "w", "m", "y", or None.

    Returns:
        List of dicts with keys: title, url, content (snippet), and raw_content (optional).
    """
    out: List[Dict[str, Any]] = []

    with DDGS() as ddgs:
        if topic == "news":
            results = list(ddgs.news(query, max_results=max_results, region=region, timelimit=timelimit))
        else:
            results = list(ddgs.text(query, max_results=max_results, region=region, timelimit=timelimit))

    for r in results:
        url = r.get("href") or r.get("url")
        snippet = r.get("body") or r.get("snippet")

        item: Dict[str, Any] = {"title": r.get("title"), "url": url, "content": snippet}

        if include_raw_content and url:
            downloaded = trafilatura.fetch_url(url)
            item["raw_content"] = trafilatura.extract(downloaded) if downloaded else None

        out.append(item)

    return out


query = "Explain the core capibilities of Deep Agents as per LangChain"
res = internet_search(query, max_results=3, topic="general", include_raw_content=True)

for i, r in enumerate(res, start=1):
    print("\n" + "=" * 80)
    print(f"[RESULT {i}]")
    print(f"[TITLE]   {r.get('title')}")
    print(f"[URL]     {r.get('url')}")
    print(f"[SNIPPET] {r.get('content')}")

    raw = r.get("raw_content")
    if raw:
        raw = raw.strip()
        preview = raw[:1200] + ("..." if len(raw) > 1200 else "")
        print("[RAW_CONTENT_PREVIEW]")
        print(preview)
    else:
        print("[RAW_CONTENT_PREVIEW] None")

print("\n" + "=" * 80)

research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`
Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

# Use the model you actually want
model = ChatOllama(model= "llama3.2:1b", #"granite4:350m", #
                   temperature=0.0)

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
    model=model,
    debug=True,  # DeepAgents debug flag
)

result = agent.invoke({"messages": [{"role": "user", 
                                     "content": query}
                                     ]})
print("\n" + "=" * 80)
print(result["messages"][-1].content)
