```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	memory(memory)
	generator(generator)
	tool_caller(tool_caller)
	reflect(reflect)
	save_chat(save_chat)
	__end__([<p>__end__</p>]):::last
	__start__ --> memory;
	generator -.-> reflect;
	generator -.-> tool_caller;
	memory --> generator;
	reflect -. &nbsp;end_without_save&nbsp; .-> __end__;
	reflect -.-> generator;
	reflect -. &nbsp;save&nbsp; .-> save_chat;
	tool_caller --> generator;
	save_chat --> __end__;
	generator -.-> generator;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```