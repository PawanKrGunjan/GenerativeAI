#!/usr/bin/env python3
from __future__ import annotations

import re
import json
from typing import Literal, Optional, List

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from logger_config import setup_logger


# =========================
# Logging
# =========================
logger = setup_logger(debug_mode=True, log_name="simple-llm-calc", log_dir="logs")
logger.info("Logger configured")


# =========================
# Output schema (structured)
# =========================
OperationType = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "modulus",
    "floor_divide",
    "square_root",
    "absolute",
]


class CalculatorOutput(BaseModel):
    result: float
    operation_performed: OperationType
    numbers: List[float] = Field(default_factory=list)
    note: str = ""


# =========================
# Deterministic parsing
# =========================
_num_re = re.compile(r"(-?\d+(?:\.\d+)?)")

def extract_numbers(text: str) -> List[float]:
    return [float(x) for x in _num_re.findall(text)]

def detect_operation(q: str) -> OperationType:
    s = q.lower()

    if "square root" in s or "sqrt" in s:
        return "square_root"
    if "absolute" in s or "abs(" in s or "absolute value" in s:
        return "absolute"
    if "floor divide" in s or "//" in s:
        return "floor_divide"
    if "mod" in s or "modulus" in s or "%" in s:
        return "modulus"
    if "power" in s or "**" in s or "to the power" in s:
        return "power"
    if "divide" in s or "/" in s:
        return "divide"
    if "multiply" in s or "times" in s or "*" in s:
        return "multiply"
    if "subtract" in s or "minus" in s:
        return "subtract"
    return "add"


def compute(op: OperationType, nums: List[float], original: str) -> CalculatorOutput:
    # Unary ops
    if op in ("square_root", "absolute"):
        if not nums:
            raise ValueError("No number found in query.")
        a = nums[0]
        if op == "square_root":
            if a < 0:
                raise ValueError("Square root of negative number is not supported (real domain).")
            return CalculatorOutput(result=a ** 0.5, operation_performed=op, numbers=nums)
        return CalculatorOutput(result=abs(a), operation_performed=op, numbers=nums)

    # Binary ops: need 2 numbers
    if len(nums) < 2:
        raise ValueError("Need two numbers for this operation.")

    # Special language case: "Subtract 13 from 50" means 50 - 13
    s = original.lower()
    if op == "subtract" and "from" in s:
        a, b = nums[1], nums[0]
    else:
        a, b = nums[0], nums[1]

    if op == "add":
        r = a + b
    elif op == "subtract":
        r = a - b
    elif op == "multiply":
        r = a * b
    elif op == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        r = a / b
    elif op == "power":
        r = a ** b
    elif op == "modulus":
        if b == 0:
            raise ValueError("Modulus by zero is not allowed.")
        r = a % b
    elif op == "floor_divide":
        if b == 0:
            raise ValueError("Floor division by zero is not allowed.")
        r = a // b
    else:
        raise ValueError(f"Unsupported operation: {op}")

    return CalculatorOutput(result=float(r), operation_performed=op, numbers=nums)


# =========================
# Tool: only requires "query"
# =========================
@tool
def calculator(query: str) -> str:
    """
    Calculator tool.
    Input: a natural-language query string.
    Output: JSON string of CalculatorOutput.
    """
    logger.info("TOOL calculator | query=%s", query)

    nums = extract_numbers(query)
    op = detect_operation(query)
    out = compute(op, nums, query)

    logger.info("TOOL calculator | op=%s nums=%s result=%s", out.operation_performed, out.numbers, out.result)
    return out.model_dump_json()


# =========================
# LLM tool calling
# =========================
MODEL_NAME = "granite4:350m"
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)
llm_tools = llm.bind_tools([calculator])


def run_calculation(query: str) -> CalculatorOutput:
    logger.info("RUN | query=%s", query)

    msg = llm_tools.invoke(
        [
            ("system", "You are a math assistant. Always use the calculator tool. "
                       "Call calculator with a single argument: the user's query string."),
            ("human", query),
        ]
    )

    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        raise ValueError(f"LLM did not call calculator. Model said: {msg.content}")

    # Execute first tool call (you can extend to handle multiple)
    tc = tool_calls[0]
    if tc.get("name") != "calculator":
        raise ValueError(f"Unexpected tool call: {tc.get('name')}")

    args = tc.get("args", {}) or {}
    tool_result_json = calculator.invoke(args)  # JSON string
    return CalculatorOutput.model_validate_json(tool_result_json)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    examples = [
        "What is 15 plus 27?",
        "Subtract 13 from 50",
        "Multiply 8 by 9",
        "Divide 100 by 4",
        "What is 5 to the power of 3?",
        "What is 17 mod 5?",
        "Floor divide 20 by 6",
        "Square root of 144",
        "Absolute value of -42",
    ]

    for q in examples:
        print("\nQuery:", q)
        try:
            out = run_calculation(q)
            print("Result:", out.result, f"({out.operation_performed})")
            print(out.model_dump_json(indent=2))
        except Exception as e:
            print("Error:", e)
        print("=" * 90)
