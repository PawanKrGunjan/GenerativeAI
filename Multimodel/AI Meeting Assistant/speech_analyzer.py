"""
AI Meeting Assistant
====================
A professional Gradio-based application for transcribing meeting audio,
normalizing financial terminology, and generating structured meeting minutes with tasks.

Requirements:
- torch, gradio>=4.0, langchain-ollama, langchain-core, transformers
- Ollama running with 'qwen2.5:3b' model
"""

import os
import torch
import gradio as gr
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Transformers for speech recognition
from transformers import pipeline


@dataclass
class MeetingOutput:
    """Structured output containing minutes and file path."""
    minutes: str
    output_path: Optional[str]


class MeetingAssistant:
    """Main class for the AI Meeting Assistant application."""
    
    def __init__(self, model_name: str = "qwen2.5:3b"):
        """Initialize the meeting assistant with Ollama LLM."""
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.5,
            num_ctx=4096  # Increased context for meeting transcripts
        )
        self.setup_chains()
    
    def setup_chains(self) -> None:
        """Initialize LangChain components."""
        # Financial terminology normalization chain
        self.financial_normalizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial terminology expert. Normalize acronyms and spoken financial terms in meeting transcripts.

RULES:
1. Expand common financial acronyms: ROA → Return on Assets (ROA)
2. Convert spoken numbers: "four zero one k" → "401(k) Retirement Savings Plan"
3. Preserve percentages and general numbers unchanged
4. Disambiguate context-sensitive terms (LTV, EBITDA, etc.)
5. Return ONLY the corrected transcript - no explanations

Output format: Clean transcript text only."""),
            ("user", "{transcript}")
        ])
        
        self.normalizer_chain = (
            self.financial_normalizer_prompt | self.llm | StrOutputParser()
        )
        
        # Meeting minutes generation chain
        self.meeting_prompt = ChatPromptTemplate.from_template("""
Generate professional meeting minutes and task list from the transcript.

CONTEXT:
{context}

OUTPUT FORMAT:
## MEETING MINUTES
### Key Discussion Points
- Bullet point summary

### Decisions Made
- Clear action outcomes

### TASK ASSIGNMENTS
| Task | Assignee | Deadline | Priority |
|------|----------|----------|----------|
| Description | Name | YYYY-MM-DD | High/Medium/Low |
""")
        
        self.meeting_chain = (
            {"context": RunnablePassthrough()} | self.meeting_prompt | self.llm | StrOutputParser()
        )
    
    @staticmethod
    def clean_ascii_text(text: str) -> str:
        """Remove non-ASCII characters while preserving essential content."""
        return ''.join(char for char in text if ord(char) < 128 or char.isspace())
    
    def normalize_financial_terms(self, transcript: str) -> str:
        """Normalize financial terminology in transcript."""
        return self.normalizer_chain.invoke({"transcript": transcript})
    
    def generate_meeting_minutes(self, processed_transcript: str) -> str:
        """Generate structured meeting minutes and tasks."""
        return self.meeting_chain.invoke({"context": processed_transcript})

    def process_audio(self, audio_file: str) -> MeetingOutput:
        """Complete audio processing pipeline with CPU-safe Whisper."""
        try:
            stt_pipeline = pipeline(
                task="automatic-speech-recognition",
                model="openai/whisper-tiny",
                device="cpu",
                dtype=torch.float32,
            )

            raw_transcript = stt_pipeline(
                audio_file,
                return_timestamps=True,   # ✅ REQUIRED for long-form
                generate_kwargs={
                    "task": "transcribe",
                    "language": "en",
                }
            )["text"]


            ascii_transcript = self.clean_ascii_text(raw_transcript)
            normalized_transcript = self.normalize_financial_terms(ascii_transcript)
            meeting_minutes = self.generate_meeting_minutes(normalized_transcript)

            timestamp = Path(audio_file).stem
            output_path = Path("outputs") / f"meeting_minutes_{timestamp}.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Meeting Minutes – {timestamp}\n\n")
                f.write(meeting_minutes)

            return MeetingOutput(
                minutes=meeting_minutes,
                output_path=str(output_path)
            )

        except Exception as e:
            return MeetingOutput(
                minutes=f"❌ Audio Processing Error:\n\n{str(e)}",
                output_path=None
            )


def create_gradio_interface():
    """Create professional Gradio interface compatible with Gradio 4.x."""
    assistant = MeetingAssistant()
    
    with gr.Blocks(title="AI Meeting Assistant") as demo:
        gr.Markdown("""
        # 🤖 AI Meeting Assistant
        **Professional audio transcription, financial term normalization, and meeting minutes generation.**
        
        *Powered by Ollama (qwen2.5:3b) + Whisper STT*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="📁 Upload Meeting Audio"
                )
                
                process_btn = gr.Button("🚀 Process Meeting", variant="primary")
            
            with gr.Column(scale=2):
                output_text = gr.Markdown(
                    label="📋 Meeting Minutes & Tasks",
                    value="👆 Upload audio and click Process to generate minutes and tasks."
                )
                
                download_file = gr.File(label="💾 Download Full Report")
        
        # Event handlers
        def handle_audio(audio_file):
            if not audio_file:
                return "⚠️ Please upload an audio file.", None
            
            result = assistant.process_audio(audio_file)
            return result.minutes, result.output_path
        
        process_btn.click(
            fn=handle_audio,
            inputs=[audio_input],
            outputs=[output_text, download_file]
        )
        
        gr.Markdown("""
        ---
        ## ✨ Features
        - **Speech-to-Text**: Whisper Tiny (CPU-optimized, English)
        - **Financial Normalization**: Auto-expands acronyms (ROA → Return on Assets)
        - **Meeting Intelligence**: Structured minutes + task assignments table
        - **Export**: Professional Markdown reports
        - **Supported Formats**: MP3, WAV, M4A, FLAC (max ~30 min)
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),  # Moved to launch()
        # css moved to launch() in Gradio 6.0, but compatible with 4.x
    )
