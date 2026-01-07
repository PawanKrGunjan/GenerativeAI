"""
Complete Code Generation Project with 3-Agent System:
- Manager (Human): Gives tasks, approves actions
- Developer: Writes code, asks clarifications  
- Tester: Writes tests, finds bugs
"""

from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig
import os
from dotenv import load_dotenv
import subprocess
import sys

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

llm_config = LLMConfig(config_list[-1])

# ========================================
# SYSTEM MESSAGES FOR EACH AGENT
# ========================================

MANAGER_SYSTEM_MESSAGE = """
You are the PROJECT MANAGER. Your role:
1. Receive project requirements from human
2. Break down tasks and assign to Developer/Tester
3. APPROVE ALL file operations (read/write/execute) from Developer
4. Ask clarifying questions when needed
5. Coordinate between Developer and Tester
6. Ask Tester for test cases when appropriate
7. Give short YES/NO feedback or brief instructions
8. Say "PROJECT COMPLETE" when done

Always respond quickly with: YES, NO, or brief instruction.
"""

DEVELOPER_SYSTEM_MESSAGE = """
You are the DEVELOPER. Your role:
1. Receive tasks from Manager
2. Ask clarifying questions if needed
3. Write clean Python code for within a Project Direcory
4. Before ANY file operation (read/write/execute), ask Manager: 
   "REQUEST: [describe action]. Approve? (YES/NO)"
5. Show the filesname with path```
6. Report status: "STATUS: Task X complete"
7. Wait for Manager approval before proceeding

NEVER execute code without Manager approval.
"""

TESTER_SYSTEM_MESSAGE = """
You are the TESTER. Your role:
1. First wait till the developer handover
Receive code from Developer via Manager
2. Write test cases
3. Find bugs and report: "BUG: [description]"
4. Before writing test files, ask Manager approval
5. Suggest fixes: "SUGGESTION: [fix description]"
6. Report: "TESTS PASSED" or "TESTS FAILED"

Wait for Manager instructions before testing.
"""

# ========================================
# AGENTS SETUP
# ========================================

def create_agents():
    # Manager (uses LLM but controlled by human)
    manager = ConversableAgent(
        name="Manager",
        system_message=MANAGER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=1,
    )
    
    # Developer Agent
    developer = ConversableAgent(
        name="Developer",
        system_message=DEVELOPER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        max_consecutive_auto_reply=5,
        description="Handles all coding tasks",
    )
    
    # Tester Agent  
    tester = ConversableAgent(
        name="Tester",
        system_message=TESTER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        max_consecutive_auto_reply=5,
        description="Handles all testing and bug finding",
    )
    
    return manager, developer, tester

# ========================================
# MAIN PROJECT LOOP
# ========================================

def main():
    print("🚀 RESUME CODE GENERATION PROJECT STARTED")
    print("=" * 60)
    print("📋 AGENTS READY: Manager (You), Developer, Tester")
    print("💻 Project: Build complete resume generator")
    print("📝 Type your project requirements below")
    print("🛑 Type 'exit' to quit")
    print("=" * 60)
    
    manager, developer, tester = create_agents()
    
    # Create Group Chat
    groupchat = GroupChat(
        agents=[manager, developer, tester],
        messages=[],
        max_round=20,
        speaker_selection_method="round_robin",  # Fair turn-taking
    )
    
    group_manager = GroupChatManager(
        groupchat=groupchat,
        name="Project_Lead",
        llm_config=llm_config,
        human_input_mode="NEVER",  # Manager handles human input
    )
    
    print("\n🎯 SCHEDULED MEETING: Full Team Briefing")
    print("📢 Manager, please start by describing the resume project requirements:\n")
    
    # Start the conversation
    try:
        group_manager.initiate_chat(
            manager,
            message="🎯 TEAM MEETING START\n\n"
                   "📋 Project: Build a complete resume generator system\n"
                   "👥 Team: Manager (Human), Developer, Tester\n"
                   "🎯 Manager: Please provide detailed project requirements.\n\n"
                   "Human input required now:"
        )
    except KeyboardInterrupt:
        print("\n👋 Project terminated by user")
    
    print("\n✅ PROJECT SESSION COMPLETE")

if __name__ == "__main__":
    main()
