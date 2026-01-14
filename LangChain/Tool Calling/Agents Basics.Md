# When & How to Use AI Agents  
Study Notes & Decision Guide

## Overview

Practical notes covering:
- When AI agents are (and are **not**) the right choice
- Four-criteria decision framework
- Current agent limitations & risk management
- Popular **LangChain** built-in tools & toolkits (as of 2025–2026)

Rephrased from course materials for clarity and quick reference.

## Table of Contents

- [Why Evaluate Before Using Agents?](#why-evaluate-before-using-agents)
- [AI System Complexity Spectrum](#ai-system-complexity-spectrum)
- [Four-Criteria Decision Framework](#four-criteria-decision-framework)
- [Current Limitations of AI Agents](#current-limitations-of-ai-agents)
- [When NOT to Use Agents](#when-not-to-use-agents)
- [Risk Management Guidelines](#risk-management-guidelines)
- [Future Improvements (Expected)](#future-improvements-expected)
- [Popular Built-in Tools & Toolkits in LangChain](#popular-built-in-tools--toolkits-in-langchain)
- [License](#license)

## Why Evaluate Before Using Agents?

AI agents are powerful but **not always the best solution**.  
They offer flexibility and autonomy, but come with higher cost, unpredictability, and risk.  
The key is to match the tool to the problem — reserve agents for cases where simpler approaches (prompts, chains, workflows) are insufficient.

**Main Learning Goals**
- Position agents correctly on the AI complexity spectrum
- Apply a structured 4-criteria framework to decide whether to use/build an agent
- Recognize clear cases where agents are **not suitable**
- Understand current limitations of agent technology
- Use practical risk management strategies
- Know the core components of effective agent architecture

## AI System Complexity Spectrum

| Level                  | Description                                          | Best Use Cases                                   | Cost / Predictability    |
|------------------------|------------------------------------------------------|--------------------------------------------------|--------------------------|
| Simple AI Features     | Single-purpose: classification, summarization, generation | Fast, repetitive, well-defined tasks             | Low & predictable        |
| Orchestrated Workflows | Fixed multi-step logic combining models + tools      | Structured, rule-based processes                 | Medium & predictable     |
| Autonomous Agents      | Independent decision-making, dynamic adaptation      | Open-ended, exploratory, strategic, creative     | High & variable          |

## Four-Criteria Decision Framework

Before building or deploying an agent, ask these four questions:

1. **Task Ambiguity**  
   → **High ambiguity** (unpredictable path, requires exploration, creativity) → Consider agent  
   → **Low ambiguity** (clear rules, repeatable steps) → Prefer workflow or chain

2. **Value vs. Cost**  
   Agents often consume **10–100× more tokens** due to planning, retries, and exploration.  
   Ask: *Does the business value / ROI justify the extra cost?*

   | Scenario                        | Recommendation          |
   |---------------------------------|-------------------------|
   | High-ROI strategic analysis     | Use agent               |
   | Routine / basic customer support| Use simple workflow     |

3. **Core Capability Check** (pre-launch testing)  
   Thoroughly test 3–5 mission-critical skills.  
   Examples:  
   - Research agent → find, filter & summarize credible sources  
   - Coding agent → write, debug, validate code reliably  
   - Support agent → classify → resolve simple → escalate correctly  
   - Analytics agent → clean data, detect anomalies, summarize trends  

   **If key tests fail → redesign or downgrade to a non-agent approach.**

4. **Consequence of Failure**  
   Evaluate:  
   - How quickly can errors be detected & corrected?  
   - What is the real-world impact? (financial, safety, reputation, legal)  
   - Is the failure reversible?  
   - Are built-in validation / correction mechanisms present?  

   → Use agents **only** when risk is low-to-medium or recoverable.

## Current Limitations of AI Agents (2025–2026)

- Inconsistent reasoning (same input → different results)  
- Unpredictable and sometimes very high token consumption  
- Tool/API fragility (agents break when integrations change)  
- Still often require human supervision in serious use cases  
- Persistent hallucinations and overconfidence

## When NOT to Use Agents

Avoid agents for:
- High-volume, low-value tasks (basic FAQ chat, simple lookups)  
- Strict real-time requirements (< 1–2 seconds)  
- Zero-tolerance domains (medicine, aviation, nuclear, high-security)  
- Heavily regulated industries requiring fully deterministic outcomes

## Risk Management Guidelines

### Core Agent Architecture (Keep It Simple)

Every reliable agent depends on three fundamentals:

1. **Environment** — the digital space the agent observes and acts in  
2. **Tools** — well-defined actions/interfaces it can use  
3. **System Prompt** — clear goals, rules, and behavioral boundaries

**Rule of thumb**: Start minimal. Add memory, planning loops, or multi-agent logic only after proving basic reliability.

### Risk-based Deployment Strategy

| Risk Level                  | Recommended Controls                                      |
|-----------------------------|-------------------------------------------------------------------|
| High-stakes + hard to detect| Mandatory human review + multiple validation layers               |
| High-stakes + visible       | Strong automated checks + real-time oversight                     |
| Low-stakes                  | Logging + user feedback + lightweight monitoring                  |

### Best Practices for Safe Early Deployment

- Start with **read-only** tool access  
- Require **human approval** for write/dangerous actions  
- Use **staged rollout** (internal → beta → full production)  
- Enable **comprehensive logging** and tracing

### Responsible Implementation Roadmap

1. **Proof of Concept** — low-risk, fully reversible tasks  
2. **Supervised Pilot** — moderate-risk tasks with close monitoring  
3. **Controlled Production** — scale only after proven safety & performance

## Future Improvements (Expected)

- More consistent reasoning and chain-of-thought  
- Leaner, more cost-efficient architectures  
- Better built-in error detection and self-correction  
- Stronger monitoring and observability tools  

Even with advances, **thoughtful evaluation and risk controls remain essential**.

## Popular Built-in Tools & Toolkits in LangChain

### Quick Summary

- **Tool** = single function the LLM can call (model-friendly input/output)  
- **Toolkit** = group of related tools used together  

**Important**: Always verify current availability, API keys, and pricing in the official LangChain documentation (many tools are third-party and paid).

### Main Categories & Popular Tools (2025–2026)

**1. Search & Knowledge**  
- SerpAPI → general web search  
- Google Serper → Google results (URLs + snippets)  
- Tavily Search → AI-optimized search (often highest quality for agents)  
- Wikipedia → clean knowledge-base lookups  
- ArXiv → scientific papers  

**2. Code & Data Analysis**  
- Python REPL → execute Python code  
- Pandas DataFrame → table manipulation  
- SQL Database Toolkit → natural language → SQL  
- LLMMath → solve math via Python evaluation  
- JSON Toolkit → handle large nested structures  

**3. Web & Browser**  
- Requests → HTTP/API calls  
- Playwright Browser → full browser automation (click, type, scroll…)  
- MultiOn → interact with web apps like a human  

**4. Productivity / SaaS**  
- Gmail → read/send emails  
- Office365 / Outlook  
- Slack → messaging in channels & DMs  
- GitHub → repositories, issues, PRs  
- Google Calendar → event management  

**5. Files & Documents**  
- File System → local read/write  
- Google Drive  
- Document Loaders → PDF, DOCX, Markdown…  
- VectorStore QA → RAG-style document querying  

**6. Finance & Business**  
- Yahoo Finance → market/news data  
- Polygon → real-time/historical stock, options data  
- Stripe → payments & subscriptions  

**7. Media & Generation**  
- DALL·E → text-to-image  
- HuggingFace Hub → open ML models  
- Google Imagen → image generation  

**Powerful combo tip**: Tavily Search + Python REPL + Browser tools is one of the most effective stacks for many agent use cases in 2025–2026.

## License

MIT – feel free to use, modify, and share.

**Last updated**: January 2026