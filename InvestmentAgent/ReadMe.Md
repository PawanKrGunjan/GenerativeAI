# 📈 InvestmentAgent – AI Stock Market Advisor

An **enterprise-grade AI agent system** for analyzing Indian stock market equities (NSE & BSE) using **LLMs, LangGraph, and tool-based reasoning**.

The system autonomously gathers stock information, performs structured reasoning, and generates **disciplined investment signals (BUY / HOLD / SELL)** while strictly avoiding hallucinated financial data.

---

# 🚀 Overview

InvestmentAgent is an **LLM-powered autonomous agent** designed to:

* Identify companies mentioned in user queries
* Resolve their NSE/BSE ticker symbols
* Fetch real-time stock information
* Analyze price data
* Provide structured investment recommendations

The system is built with a **LangGraph agent architecture**, ensuring deterministic workflows, tool validation, and robust error handling.

This project demonstrates **production-level agent design** for financial analysis applications.

---

# 🧠 Key Features

### AI Agent Capabilities

✔ Autonomous multi-step reasoning
✔ Tool-based stock data retrieval
✔ Symbol resolution for Indian equities
✔ Strict hallucination prevention
✔ Structured investment advice generation
✔ Reflection-based memory learning

---

### Engineering Features

✔ **LangGraph agent workflow**
✔ **Tool registry architecture**
✔ **Memory system with embeddings**
✔ **Structured logging system**
✔ **Enterprise-grade prompt safety**
✔ **Loop detection & recovery logic**

---

# 🏗 System Architecture

```
User Query
    │
    ▼
LangGraph Agent
    │
    ├── Company Identification
    │
    ├── Symbol Resolution Tool
    │
    ├── Market Data Tool
    │
    ▼
LLM Reasoning Engine
    │
    ▼
Structured Investment Advice
    │
    ▼
Memory + Reflection Storage
```

The agent uses **deterministic tool execution** to ensure all financial information is retrieved from trusted sources rather than guessed by the LLM.

---

# 📂 Project Structure

```
InvestmentAgent/
│
├── agents/
│   └── investment_agent.py
│
├── tools/
│   ├── tool_registry.py
│   ├── search_tools.py
│   └── stock_tools.py
│
├── memory/
│   └── store_memory.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── db_connect.py
│
├── api/
│   └── API endpoints for integration
│
├── web/
│   └── UI components
│
├── chat/
│   └── chat orchestration layer
│
├── analysis/
│   └── financial analysis utilities
│
├── data/
│   └── datasets and company records
│
├── graphs/
│   └── LangGraph visualizations
│
├── tests/
│   └── unit tests
│
├── logs/
│   └── runtime logs
│
├── run.py
├── terminal_run.py
└── requirements.txt
```

---

# ⚙️ Technology Stack

| Layer                | Technology                |
| -------------------- | ------------------------- |
| AI Framework         | LangGraph                 |
| LLM Interface        | LangChain                 |
| Embeddings           | Custom embedding pipeline |
| Programming Language | Python                    |
| Database             | SQL / Memory Store        |
| Logging              | Structured logging        |
| Deployment           | Docker-ready              |

---

# 🤖 Agent Workflow

The agent follows a strict multi-step process:

### Step 1 — Query Understanding

The agent analyzes the user request.

Example:

```
Should I purchase HDFC Bank share?
```

---

### Step 2 — Symbol Resolution

The agent resolves the company name to its ticker.

```
HDFC Bank → HDFCBANK.NS
```

---

### Step 3 — Market Data Retrieval

The system fetches current market data.

Example:

```
Price
Volume
Change %
Timestamp
```

---

### Step 4 — Reasoning

The LLM performs analysis using:

* retrieved market data
* historical context
* stored memory reflections

---

### Step 5 — Structured Investment Advice

The agent outputs standardized investment advice.

Example:

```
Company: HDFC Bank Ltd
Ticker: HDFCBANK.NS
Price: ₹1,602.40
Updated: 15-Mar-2026 15:30 IST

Signal: HOLD
Confidence: 74/100

Pointwise Reasoning:
• Price stabilizing after recent correction
• Strong banking fundamentals
• Volume consistent with institutional participation
• Valuation currently near sector median
```

---

# 🧩 Tools System

The agent interacts with the external world using **tools**.

### Available Tools

| Tool                | Purpose                            |
| ------------------- | ---------------------------------- |
| lookup_stock_symbol | Resolve company name to NSE symbol |
| get_stock_info      | Retrieve latest market data        |
| memory reflection   | Store analysis insights            |

Tools ensure the agent **never fabricates financial data**.

---

# 🧠 Memory System

The agent maintains **reflection memory** to store previous analyses.

Memory includes:

```
timestamp
company
symbol
previous advice
analysis summary
```

This enables:

* contextual reasoning
* pattern recognition
* historical insight

---

# 📊 Graph Visualization

LangGraph automatically generates workflow diagrams.

Location:

```
graphs/investment_agent_optimized.png
```

This helps developers visualize agent logic.

---

# 🖥 Running the Project

## 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 2️⃣ Run the agent (terminal mode)

```
python agents/investment_agent.py
```

Example interaction:

```
You: Should I buy Infosys stock?

Advisor:
Company: Infosys Ltd
Ticker: INFY.NS
Price: ₹1,512
Signal: HOLD
Confidence: 68/100
```

---

## 3️⃣ Run application

```
python run.py
```

---

# 📜 Logging

Logs are stored in:

```
logs/
```

Example log output:

```
NODE → REACT (attempt 1)
Executing tool: lookup_stock_symbol
Stored price data for HDFCBANK.NS
Final advice generated successfully
```

---

# 🔒 Safety & Guardrails

The system enforces strict rules:

* No hallucinated prices
* No guessed ticker symbols
* Mandatory tool validation
* Structured investment output
* Loop detection to prevent infinite reasoning

---

# 🧪 Testing

Run tests with:

```
pytest tests/
```

---

# 🐳 Docker Support

A Docker environment can be built using:

```
docker build -t investment-agent .
```

Run container:

```
docker run investment-agent
```

---

# 📈 Future Improvements

Planned enhancements:

* real-time market data streaming
* technical indicator tools
* portfolio optimization agent
* risk scoring model
* multi-agent research system
* backtesting engine

---

# 🤝 Contributing

Contributions are welcome.

Please follow:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

# 📄 License

MIT License

---

# 👨‍💻 Author

**Pawan Kumar Gunjan**

AI Engineer | Generative AI Developer

---

# ⭐ Acknowledgements

Built using:

* LangGraph
* LangChain
* Python AI ecosystem
* Open-source LLM tooling
