import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self, model_path=None):
        # model_path is unused now, but kept so your RAGSystem constructor still works
        self.client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        self.model = "sonar-pro"

    def generate(self, question, chunks, max_tokens=512, temperature=0.7):
        # Build context string from chunks
        context_text = "\n\n".join(
            f"[Chunk {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(chunks)
        )

        prompt = f"""
You are an expert assistant.

Use ONLY the following retrieved context to answer the question:

{context_text}

Question: {question}

Answer clearly and concisely.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "query": question,
            "answer": response.choices[0].message.content,
            "sources": [chunk.get("source", "Unknown") for chunk in chunks]
        }
