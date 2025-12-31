from __future__ import annotations

import json
from typing import Literal, Union

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama


# ==========================
# Calculator Tool Definition
# ==========================

OperationType = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",          # a ** b
    "modulus",        # a % b
    "floor_divide",   # a // b
    "square_root",    # sqrt(a)  (b is ignored)
    "absolute",       # abs(a)   (b is ignored)
]


class CalculatorInput(BaseModel):
    """Input schema for all calculator operations."""
    operation: OperationType = Field(..., description="The mathematical operation to perform")
    a: float = Field(..., description="First operand (main number)")
    b: float = Field(default=0.0, description="Second operand (used only when required)")


class CalculatorOutput(BaseModel):
    """Standardized output format."""
    result: float
    operation_performed: str
    input_used: CalculatorInput


# =====================
# Calculator Engine Class
# =====================

class MathCalculator:
    """A clean, extensible calculator that handles dispatching based on validated input."""

    @staticmethod
    def _add(a: float, b: float) -> float:
        return a + b

    @staticmethod
    def _subtract(a: float, b: float) -> float:
        return a - b

    @staticmethod
    def _multiply(a: float, b: float) -> float:
        return a * b

    @staticmethod
    def _divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b

    @staticmethod
    def _power(a: float, b: float) -> float:
        return a ** b

    @staticmethod
    def _modulus(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Modulus by zero is not allowed")
        return a % b

    @staticmethod
    def _floor_divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Floor division by zero is not allowed")
        return a // b

    @staticmethod
    def _square_root(a: float, _: float) -> float:
        if a < 0:
            raise ValueError("Square root of negative number is not supported in real domain")
        return a ** 0.5

    @staticmethod
    def _absolute(a: float, _: float) -> float:
        return abs(a)

    @classmethod
    def execute(cls, input_data: CalculatorInput) -> CalculatorOutput:
        """Dispatch to the correct operation and return structured output."""
        handlers = {
            "add": cls._add,
            "subtract": cls._subtract,
            "multiply": cls._multiply,
            "divide": cls._divide,
            "power": cls._power,
            "modulus": cls._modulus,
            "floor_divide": cls._floor_divide,
            "square_root": cls._square_root,
            "absolute": cls._absolute,
        }

        handler = handlers[input_data.operation]
        result = handler(input_data.a, input_data.b)

        return CalculatorOutput(
            result=result,
            operation_performed=input_data.operation,
            input_used=input_data,
        )


# =========================
# LLM Setup with Tool Binding
# =========================

model_name = "qwen2.5:3b"  # You can change this

llm = ChatOllama(
    model=model_name,
    temperature=0.0,  # Low temperature for reliable tool calling
)

# Bind the single tool schema to the LLM
calculator_llm = llm.bind_tools(tools=[CalculatorInput], tool_choice="auto")


# =====================
# Helper Execution Function
# =====================

def run_calculation(query: str) -> CalculatorOutput:
    """Run a natural language query through the LLM and execute the calculator."""
    response: AIMessage = calculator_llm.invoke(query)

    if not response.tool_calls:
        raise ValueError("LLM did not call the calculator tool. Response: " + response.content)

    tool_call = response.tool_calls[0]
    args = tool_call["args"]

    # Validate and parse input
    try:
        calc_input = CalculatorInput.model_validate(args)
    except ValidationError as e:
        raise ValueError(f"Invalid arguments from LLM: {e}")

    # Execute the operation
    return MathCalculator.execute(calc_input)


# =====================
# Example Usage
# =====================

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
        print(f"\nQuery: {q}")
        try:
            output = run_calculation(q)
            print(f"Result: {output.result}  ({output.operation_performed})")
            print(f"Full output: {output.model_dump_json(indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
        print('=' * 100)