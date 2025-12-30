## Starter code: provide your solutions in the TODO parts
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


# First initialize your LLM
model_id = "llama3.2"  # Or other models available in Ollama (e.g., "llama3.2:1b", "phi3", "mistral")


default_params = {'decoding_method': 'sample',
 'length_penalty': {'decay_factor': 2.5, 'start_index': 5},
 'temperature': 0.5,
 'top_p': 0.2,
 'top_k': 1,
 'random_seed': 33,
 'repetition_penalty': 2,
 'min_new_tokens': 50,
 'max_new_tokens': 200,
 'stop_sequences': ['fail'],
 'time_limit': 600000,
 'truncate_input_tokens': 200,
 'prompt_variables': {'object': 'brain'},
 'return_options': {'input_text': True,
  'generated_tokens': True,
  'input_tokens': True,
  'token_logprobs': True,
  'token_ranks': False,
  'top_n_tokens': False}}


# TODO: Initialize your LLM (locally on CPU)
llm = OllamaLLM(
    model=model_id,
    temperature=default_params['temperature'],
    top_p=default_params['top_p'],
    top_k=default_params['top_k'],
    repeat_penalty=default_params['repetition_penalty'],
    num_predict=default_params['max_new_tokens'],
    seed=default_params['random_seed'],
    stop=default_params['stop_sequences'],
)


# Here is an example template you can use
template = """
Analyze the following product review:
"{review}"


Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""


# TODO: Create your prompt template
product_review_prompt = PromptTemplate.from_template(template)


# TODO: Create a formatting function
def format_review_prompt(variables):
    return product_review_prompt.format(**variables)


# TODO: Build your LCEL chain
review_analysis_chain = (
    RunnableLambda(format_review_prompt) 
    | llm 
    | StrOutputParser()
)


# Example reviews to process
reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]


# TODO: Process the reviews
for review in reviews:
    result = review_analysis_chain.invoke({"review": review})
    print(f"Review: {review}\n")
    print(f"Analysis:\n{result}\n")
    print("-" * 80)
