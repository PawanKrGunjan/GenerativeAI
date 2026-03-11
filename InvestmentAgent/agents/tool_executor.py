from tools.tool_registry import SEARCH_TOOLS, MARKET_TOOLS, ANALYSIS_TOOLS
from utils.logger import LOGGER


class ToolExecutor:

    def __init__(self, tool_type="data"):

        if tool_type == "data":
            tools = SEARCH_TOOLS + MARKET_TOOLS
        else:
            tools = ANALYSIS_TOOLS

        self.tool_map = {tool.name: tool for tool in tools}

    def run(self, state):

        LOGGER.info("Tool Executor started")

        tool_calls = state.get("tool_calls", [])
        results = []

        for call in tool_calls:

            name = call["name"]
            args = call.get("args", {})

            tool = self.tool_map.get(name)

            try:

                # Handle multi-symbol inputs
                symbol = args.get("symbol")

                if isinstance(symbol, list):

                    for s in symbol:

                        new_args = dict(args)
                        new_args["symbol"] = s

                        LOGGER.info("Executing tool: %s (%s)", name, s)

                        result = tool.invoke(new_args)

                        results.append(result)

                else:

                    LOGGER.info("Executing tool: %s", name)

                    result = tool.invoke(args)

                    results.append(result)

            except Exception:

                LOGGER.exception("Tool failed: %s", name)

        state["tool_results"] = results

        return state