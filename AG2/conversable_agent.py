# 1. Import ConversableAgent class
from autogen import ConversableAgent, LLMConfig
from dotenv import load_dotenv
import os
load_dotenv()

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



# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig(config_list[0])
print(llm_config)

# 3. Create our LLM agent
# Create an AI agent
assistant = ConversableAgent(
    name="assistant",
    system_message="You are an assistant that responds concisely.",
    llm_config=llm_config
)

# Create another AI agent
fact_checker = ConversableAgent(
    name="fact_checker",
    system_message="You are a fact-checking assistant.",
    llm_config=llm_config
)

# 4. Start the conversation
assistant.initiate_chat(
    recipient=fact_checker,
    message="What is AG2?",
    max_turns=2
)

# # 4. Run the agent with a prompt
# response = my_agent.run(
#     message="In one sentence, what's the big deal about AI?",
#     max_turns=3,
# )

# # 5. Iterate through the chat automatically with console output
# response.process()

# # 6. Print the chat
# print(response.messages)