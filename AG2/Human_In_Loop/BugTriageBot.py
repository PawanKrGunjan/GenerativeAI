from autogen import ConversableAgent, LLMConfig
import os
import random
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

llm_config = LLMConfig(config_list[0])
#print(llm_config)

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

# Step 3: Create the assistant agent
with llm_config:
    triage_bot = ConversableAgent(
        name="triage_bot",
        system_message=triage_system_message,
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
response = human.run(
    recipient=triage_bot,
    message=initial_prompt,
)

# Step 7: Display the response
response.process()
