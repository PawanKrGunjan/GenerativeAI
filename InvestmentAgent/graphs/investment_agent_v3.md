```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	resolve_symbols(resolve_symbols)
	load_memory(load_memory)
	agent(agent)
	tools(tools)
	indicator_cache(indicator_cache)
	sentiment(sentiment)
	reflect(reflect)
	memory_update(memory_update)
	__end__([<p>__end__</p>]):::last
	__start__ --> resolve_symbols;
	agent -.-> __end__;
	agent -.-> reflect;
	agent -.-> tools;
	indicator_cache --> sentiment;
	load_memory --> agent;
	reflect --> memory_update;
	resolve_symbols --> load_memory;
	sentiment --> agent;
	tools --> indicator_cache;
	memory_update --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```