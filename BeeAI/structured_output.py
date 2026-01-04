import asyncio
import logging
from pydantic import BaseModel, Field
from typing import List
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")
print("Key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Base set:", os.getenv("OPENAI_API_BASE"))

# Define a structured output for business planning
class BusinessPlan(BaseModel):
    """A comprehensive business plan structure."""
    business_name: str = Field(description="Catchy name for the business")
    elevator_pitch: str = Field(description="30-second description of the business")
    target_market: str = Field(description="Primary target audience")
    unique_value_proposition: str = Field(description="What makes this business special")
    revenue_streams: List[str] = Field(description="Ways the business will make money")
    startup_costs: str = Field(description="Estimated initial investment needed")
    key_success_factors: List[str] = Field(description="Critical elements for success")

async def structured_output_example():
    #llm = ChatModel.from_name("openai:gpt-5-nano", ChatModelParameters(temperature=0))
    llm = ChatModel.from_name("openai:sonar-pro",ChatModelParameters(temperature=0))
    #llm = ChatModel.from_name("ollama:llama3.2", ChatModelParameters(temperature=0, max_tokens=300))

    messages = [
        SystemMessage(content="You are an expert business consultant and entrepreneur."),
        UserMessage(content="Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    ]
    
    # Generate structured response using create_structure() method
    response = await llm.create_structure(
        schema=BusinessPlan,
        messages=messages
    )
    
    print("User: Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    print("\n🚀 AI-Generated Business Plan:")
    print(f"💡 Business Name: {response.object['business_name']}")
    print(f"🎯 Elevator Pitch: {response.object['elevator_pitch']}")
    print(f"👥 Target Market: {response.object['target_market']}")
    print(f"⭐ Unique Value Proposition: {response.object['unique_value_proposition']}")
    print(f"💰 Revenue Streams: {', '.join(response.object['revenue_streams'])}")
    print(f"💵 Startup Costs: {response.object['startup_costs']}")
    print(f"🔑 Key Success Factors:")
    for factor in response.object['key_success_factors']:
        print(f"  - {factor}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress unwanted warnings
    await structured_output_example()

if __name__ == "__main__":
    asyncio.run(main())