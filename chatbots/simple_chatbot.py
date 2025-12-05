# chatbot.py  ←  Copy-paste this entire file and run

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()

# Perplexity LLM — Correct model name (December 2025)
llm = ChatOpenAI(
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    model="sonar-pro",           # ← THIS IS THE CORRECT MODEL
    temperature=0.7
)

print("Perplexity Chatbot is ready! (type 'quit' or 'exit' to stop)\n")

# This list keeps the full conversation history (so the bot remembers everything)
chat_history = []

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        break
    
    if user_input == "":
        continue

    # Add your message to history
    chat_history.append(HumanMessage(content=user_input))

    # Get response from Perplexity
    response = llm.invoke(chat_history)
    bot_reply = response.content

    # Add bot reply to history (for context in next turn)
    chat_history.append(AIMessage(content=bot_reply))

    print(f"Bot: {bot_reply}\n")