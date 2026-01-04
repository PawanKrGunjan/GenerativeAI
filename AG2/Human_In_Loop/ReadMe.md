### Human-in-the-Loop Example: Bug Triage Bot

This example demonstrates how to use AG2’s `ConversableAgent` in `human_input_mode="ALWAYS"` to enable **human-in-the-loop workflows**.

We simulate a **bug triage assistant** (`triage_bot`) that classifies bug reports as either:
- Escalate (for example, critical crash or security issue),
- Close (for example, minor cosmetic issue),
- Medium priority (default for others).

For each classification, the assistant **asks the human agent for confirmation or correction**. This ensures the AI doesn’t act on high-impact decisions without oversight.

At the end, the assistant summarizes the triage results.

---

### Try these inputs when prompted

When you’re prompted to reply as the human agent, try responding with the following:

- **Confirm assistant’s suggestion**  
  `"Yes, escalate it."`  
  `"Closing this makes sense."`

- **Override assistant’s suggestion**  
  `"This should be marked as high priority instead."`  
  `"Let’s keep this open for now."`

- **Ask for clarification**  
  `"Why do you think this is low priority?"`  
  `"Can you provide more reasoning?"`

You can also type `exit` at any time to end the conversation.
