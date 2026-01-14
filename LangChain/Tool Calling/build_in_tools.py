from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool, Tool
# Create a Wikipedia tool using the @tool decorator
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information about a topic.
    
    Parameters:
    - query (str): The topic or question to search for on Wikipedia
    
    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

res = search_wikipedia.invoke("What is Agentic AI?")
print(res)
