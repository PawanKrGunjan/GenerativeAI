import os
import asyncio
import logging
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")
print("Key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Base set:", os.getenv("OPENAI_API_BASE"))


async def basic_chat_example():
    # Use dict for params (avoids ChatModelParameters serialization bug in litellm adapter)
    llm = ChatModel.from_name("openai:sonar-pro")

    messages = [
        SystemMessage(content="You are a helpful AI assistant and creative writing expert."),
        UserMessage(content="Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    ]
    
    response = await llm.create(messages=messages)
    
    print("User: Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    print(f"Assistant: {response.get_text_content()}")
    
    return response

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    response = await basic_chat_example()

if __name__ == "__main__":
    asyncio.run(main())
