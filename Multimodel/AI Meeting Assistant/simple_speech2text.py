import torch
import os
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_classic.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline  # For Speech-to-Text

#######------------- LLM Initialization-------------#######
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.5,
    num_ctx=2048,
)


prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer clearly and concisely:\n{question}"
)

#######------------- Helper Functions-------------#######

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products.
Your task is to normalize financial terms in transcripts.

Rules:
- Expand acronyms: ROA → Return on Assets (ROA)
- Convert spoken numbers: "four zero one k" → "401(k) (Retirement Savings Plan)"
- Disambiguate context-sensitive acronyms (e.g., LTV)
- Do NOT change percentages or general numbers
- Return:
  1. The corrected transcript
  2. A list of changed terms
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ascii_transcript},
    ]

    response = llm.invoke(messages)
    return response.content

#######------------- Prompt Template and Chain-------------#######

# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": RunnablePassthrough()}  # Pass the transcript as context
    | prompt
    | llm
    | StrOutputParser()
)


#######------------- Speech2text and Pipeline-------------#######

# Speech-to-text pipeline
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        #model="openai/whisper-tiny.en",
        model="openai/whisper-tiny",
        #chunk_length_s=30,
        device='cpu',
        ignore_warning=True
    )

    raw_transcript = pipe(audio_file, batch_size=8)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)

    return result, output_file

res, output_file = transcript_audio('sample-meeting.wav')
print(res)