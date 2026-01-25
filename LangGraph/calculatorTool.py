import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI (works with Ollama)
mlflow.openai.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LangGraph")

# Initialize the OpenAI client with Ollama API endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    temperature=0.1,
    max_tokens=100,
)