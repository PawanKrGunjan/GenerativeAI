from llama_index.core.agent import ReActAgent
import llama_index.core
print(f"Llama Index Core Version: {llama_index.core.__version__}")
try:
    print(f"ReActAgent.from_tools exists: {hasattr(ReActAgent, 'from_tools')}")
except Exception as e:
    print(f"Error checking from_tools: {e}")

print("Dir of ReActAgent:")
print(dir(ReActAgent))
