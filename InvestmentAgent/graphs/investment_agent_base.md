```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	REACT(REACT)
	Tool_Call(Tool_Call)
	REFLECT(REFLECT)
	__end__([<p>__end__</p>]):::last
	REACT -.-> REFLECT;
	REACT -.-> Tool_Call;
	REACT -.-> __end__;
	Tool_Call --> REACT;
	__start__ --> REACT;
	REFLECT --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```