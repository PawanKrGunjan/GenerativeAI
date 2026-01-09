from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import datetime  # Added for current date

load_dotenv()

# Perplexity LLM (added api_key, removed trailing / from base_url)
perplexity_llm = LLM(
    model="sonar-pro", 
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY"),  # Required!
    temperature=0.3,
    max_tokens=2000
)



# Local Ollama LLM fallback (if needed)
ollama_llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)

search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))  # Requires SERPER_API_KEY in .env

## Agents
researcher = Agent(
    role="Content Researcher",
    goal="Discover latest trends, insights, and data on given topics for high-quality content creation.",
    backstory="Seasoned researcher skilled in finding authoritative sources, extracting key facts, and synthesizing information into actionable insights. Always verify facts with multiple sources.",
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,
)

writer = Agent(
    role="Content Writer",
    goal="Craft engaging, well-structured blog posts from research findings.",
    backstory="Expert writer who creates captivating long-form content that's informative, SEO-optimized, and reader-focused. Uses storytelling, data, and clear structure to maximize impact.",
    llm=perplexity_llm,
    verbose=True,
    allow_delegation=True
)

social_strategist = Agent(
    role="Social Media Strategist",
    goal="Curate compelling summaries and short-form content optimized for social platforms like LinkedIn and X/Twitter from research and blog content.",
    backstory="You are a top-tier social media strategist with 10+ years experience amplifying content reach. Expert at distilling complex ideas into viral-ready posts that drive engagement, shares, and conversations. Always tailor content to platform algorithms and audience behaviors.",
    tools=[search_tool],
    llm=perplexity_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=15,
)

## Tasks
research_task = Task(
    description="""Research the topic '{topic}'. 
    Find 8-10 key insights, statistics, and trends relevant to {topic}.
    Focus on current developments as of {current_date}.
    Output as bulleted list with sources.""",
    expected_output="A comprehensive research summary with 8-10 bulleted insights on {topic}, including sources.",
    agent=researcher
)

writing_task = Task(
    description="""Using the research from previous task, write a 800-1000 word blog post on '{topic}'.
    Structure: Engaging intro, 4-5 sections with subheadings, data-backed points, conclusion with CTA.
    Make it readable, authoritative, and optimized for sharing.""",
    expected_output="A complete, publication-ready blog post (800-1000 words) on {topic}.",
    agent=writer,
    context=[research_task]
)

social_task = Task(
    description="""Review the full research and blog post. 
    Create 3 short-form posts:
    1. LinkedIn post (professional, 200-300 words, with key insights + CTA).
    2. X/Twitter thread (3-5 tweets, punchy, engaging).
    3. Summary teaser (100 words for any platform).
    Optimize each for platform: hashtags, emojis, questions for engagement.""",
    expected_output="""Three platform-optimized social posts:
    - LinkedIn post
    - X/Twitter thread
    - Short summary teaser
    Ready to copy-paste and post.""",
    agent=social_strategist,
    context=[research_task, writing_task]
)

social_crew = Crew(
    agents=[researcher, writer, social_strategist],
    tasks=[research_task, writing_task, social_task],
    process=Process.sequential,
    verbose=True,
    memory=False,       # Keep enabled now that embedder is fixed
    cache=True,
    max_rpm=100,
    share_crew=False,
    tracing=True,
    # embedder={
    #     "provider": "ollama",
    #     "config": {
    #         "embeddings_ollama_model_name": "nomic-embed-text:latest",  # This exact key fixes the validation error
    #         # Optional: custom URL if Ollama isn't on localhost
    #         "ollama_base_url": "http://localhost:11434"
    #     }
    # }
)


if __name__ == "__main__":
    topic = "AI Agents in Marketing 2026"
    
    # Dynamically get current date in a nice format
    current_date = datetime.date.today().strftime("%B %d, %Y")  # e.g. "January 03, 2026"
    
    result = social_crew.kickoff(inputs={"topic": topic, "current_date": current_date})
    print("Final Output:\n", result)