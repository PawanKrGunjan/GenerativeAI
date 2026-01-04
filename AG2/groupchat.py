from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
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

# Define system messages and agent descriptions
planner_message = "Create a short lesson plan for 4th graders."
reviewer_message = "Review a plan and suggest up to 3 brief edits."
teacher_message = "Suggest a topic and reply DONE when satisfied."

with llm_config:
    lesson_planner = ConversableAgent(
        name="planner_agent",
        system_message=planner_message,
        description="Makes lesson plans.",
    )

    lesson_reviewer = ConversableAgent(
        name="reviewer_agent",
        system_message=reviewer_message,
        description="Reviews lesson plans and suggests edits.",
    )

    teacher = ConversableAgent(
        name="teacher_agent",
        system_message=teacher_message,
        is_termination_msg=lambda x: "DONE" in (x.get("content", "") or "").upper()
    )

# Configure the group chat with automatic speaker selection
groupchat = GroupChat(
    agents=[teacher, lesson_planner, lesson_reviewer],
    speaker_selection_method="auto"  # Uses AutoPattern
)

manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config
)

# Start with a short initial prompt to keep tokens low
teacher.initiate_chat(
    recipient=manager,
    message="Make a simple lesson about the moon.",
    max_turns=6,  # Limit total rounds (e.g., 2 per agent max) -  As a safeguard, it's always best to use max_turns to prevent runaway loops.
    summary_method="reflection_with_llm"
)