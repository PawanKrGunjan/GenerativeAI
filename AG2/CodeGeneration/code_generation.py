from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv
import os

load_dotenv()


# Step 1: Configure the LLM to use (e.g., GPT-4o Mini via OpenAI)
config_list = [
    {
        "model": "sonar-pro",
        "api_key": os.getenv("PERPLEXITY_API_KEY"),
        "base_url": "https://api.perplexity.ai",
        "api_type": "openai",
        "temperature": 0.3,
        "max_tokens": 1000
    },
    {
        "model": "llama3.2:latest",
        "api_type": "ollama",
        "client_host": "http://192.168.0.1:11434",
        "temperature": 0.0,
        "max_tokens": 200
    },
    {
        "model": "gemini-2.5-flash",  # Correct model name with version [web:21]
        "api_key": os.getenv("GEMINI_API_KEY"),
        "api_type": "google",
        "temperature": 0.3,
        "max_tokens": 8192  # Gemini max output limit [web:21]
    }
]


# Step 1: Configure the LLM to use (e.g., GPT-4o Mini via OpenAI)
llm_config = LLMConfig(config_list[2])
print(llm_config)

# Step 2: Create the assistant agent (code-writing AI)
with llm_config:
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant who writes and explains Python code clearly."
    )

# Step 3: Create the user agent that can execute code
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Automatically executes code without human input
    max_consecutive_auto_reply=5,  # Ends after 5 response cycles (assistant + user_proxy turns)
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="coding", timeout=30),
    },
)

# Step 4: Start a simple task that leads to code generation and execution
chat_result = user_proxy.initiate_chat(
    recipient=assistant,
    message="""Plot a sine, cose and tan wave using matplotlib in single graph with distinct color from -2π to 2π and save the plot as sine_wave.png.""",
    max_turns=4,  # 2 rounds of assistant ↔ user_proxy
    summary_method="reflection_with_llm"  # Optional: final LLM-generated summary
)

# Step 5: Display the generated figure (optional for notebook environments)
from IPython.display import Image, display
import os

image_path = "coding/graph.png"
if os.path.exists(image_path):
    display(Image(filename=image_path))
else:
    print("Plot not found. Please check if the assistant saved the file correctly.")

# Step 6: Print summary
print("\n Final Summary:")
print(chat_result.summary)
