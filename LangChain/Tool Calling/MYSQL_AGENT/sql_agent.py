# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

from langchain_ollama import ChatOllama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Load env vars
load_dotenv()

# Chat model (required for agents)
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

# MySQL creds
mysql_username = os.getenv("MYSQL_USERNAME")
mysql_password = quote_plus(os.getenv("MYSQL_PASSWORD"))
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_port = os.getenv("MYSQL_PORT", "3306")
database_name = os.getenv("MYSQL_DATABASE")

mysql_uri = (
    f"mysql+mysqlconnector://{mysql_username}:{mysql_password}"
    f"@{mysql_host}:{mysql_port}/{database_name}"
)

db = SQLDatabase.from_uri(mysql_uri)

# Create SQL agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=False,  # 👈 cleaner chat
    handle_parsing_errors=True,
)

print("\n💬 SQL Chat Agent (type 'exit' or 'quit' to stop)\n")

# =====================
# Chat loop
# =====================
while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        response = agent.invoke({"input": user_input})
        print(f"Agent: {response['output']}\n")

    except KeyboardInterrupt:
        print("\n👋 Chat terminated")
        break

    except Exception as e:
        print(f"⚠️ Error: {e}\n")
