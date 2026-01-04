from autogen import ConversableAgent, register_function, LLMConfig
from typing import Annotated
import os
from dotenv import load_dotenv
load_dotenv()


# Step 1: Configure the LLM to use (e.g., GPT-4o Mini via OpenAI)
config_list=[
    {
        "model": "sonar-pro",
        "api_key": os.getenv("PERPLEXITY_API_KEY"),
        "base_url": "https://api.perplexity.ai",
        #"api_rate_limit": 60.0,
        #"api_type": "openai",
        "temperature": 0.3,
        "max_tokens": 1000},
    {
        "model": "llama3.2:latest",
        "api_type": 'ollama',
        "client_host": "http://192.168.0.1:11434",
        "temperature": 0.0,
        "max_tokens": 200}]
# Replace with your actual key if running outside this environment
#llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")
llm_config = LLMConfig(config_list[0])


# Define a simple utility function to check if a number is prime
def is_prime(n: Annotated[int, "Positive integer"]) -> str:
    if n < 2:
        return "No"
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return "No"
    return "Yes"

# Create the asking agent and the tool-using agent
with llm_config:
    math_asker = ConversableAgent(
        name="math_asker",
        system_message="Ask whether a number is prime."
    )
    math_checker = ConversableAgent(
        name="math_checker",
        human_input_mode="NEVER"
    )

# Register the function between the two agents
register_function(
    is_prime,
    caller=math_asker,
    executor=math_checker,
    description="Check if a number is prime. Returns Yes or No."
)

# Start a brief conversation
math_checker.initiate_chat(
    recipient=math_asker,
    message="Is 72 a prime number?",
    max_turns=2
)
