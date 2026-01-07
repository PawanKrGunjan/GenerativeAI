from autogen import ConversableAgent, LLMConfig
import os
import random
from dotenv import load_dotenv

load_dotenv()

# Step 1: Configure the LLM to use (fixed model name and removed deprecated context manager)
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

llm_config = LLMConfig(config_list[2])  # Using gemini (index 1)

# Step 2: Define system message for bug triage assistant
triage_system_message = """
You are a bug triage assistant. You will be given bug report summaries.

For each bug:
- If it is urgent (e.g., 'crash', 'security', or 'data loss' is mentioned), escalate it and ask the human agent for confirmation.
- If it seems minor (e.g., cosmetic, typo), suggest closing it but still ask for human review.
- Otherwise, classify it as medium priority and ask the human for review.

Once all bugs are processed, summarize what was escalated, closed, or marked as medium priority.
End by saying: "You can type exit to finish."
"""

# Step 3: Create the assistant agent (REMOVED deprecated 'with llm_config:')
triage_bot = ConversableAgent(
    name="triage_bot",
    system_message=triage_system_message,
    llm_config=llm_config  # Pass directly - no context manager
)

# Step 4: Create the human agent who will review each recommendation
human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",  # prompts for input at each step
)

# Step 5: Generate sample bug reports
BUGS = [
    "App crashes when opening user profile.",
    "Minor UI misalignment on settings page.",
    "Password reset email not sent consistently.",
    "Typo in the About Us footer text.",
    "Database connection timeout under heavy load.",
    "Login form allows SQL injection attack.",
]

random.shuffle(BUGS)
selected_bugs = BUGS[:3]

# Format the initial task
initial_prompt = (
    "Please triage the following bug reports one by one:\n\n" +
    "\n".join([f"{i+1}. {bug}" for i, bug in enumerate(selected_bugs)])
)

# Step 6: Start the conversation
response = human.initiate_chat(
    recipient=triage_bot,
    message=initial_prompt,
)

# Step 7: No need for response.process() - initiate_chat handles it
#response.process()