import os
import re
import time
import logging
import torch
from dotenv import load_dotenv

import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
from IPython.display import Audio, display

from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger("Voicebot")
logger.info(f"PyTorch version: {torch.__version__}")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress ALSA warnings on Linux
def suppress_alsa_warnings():
    if os.name != "nt":
        try:
            import sys
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stderr.fileno())
        except Exception as e:
            logger.warning(f"Could not suppress ALSA warnings: {e}")

suppress_alsa_warnings()

# ----------------------------- Main Class -----------------------------
class VoiceChatbot:
    def __init__(self, model_path: str = r'./models/gemma-2-2b-it-Q4_K_M.gguf'):
        """
        Initialize the VoiceChatbot with LLM, speech models, and configurations.
        """
        logger.info("Initializing VoiceChatbot...")
        load_dotenv()  # Load .env if you have API keys

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Device selection
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            self.device = "cuda:0" if major >= 7 else "cpu"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")


        # Speech-to-text (Whisper)
        logger.info("Loading Whisper model for transcription...")
        device_index = 0 if self.device.startswith("cuda") else -1

        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
            device=device_index
        )

        # Wake word / command classifier (e.g., for 'stop')
        logger.info("Loading audio classifier for wake/stop detection...")
        self.classifier = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-speech-commands-v2",
            device=device_index
        )

        # Local LLM (LlamaCpp)
        logger.info(f"Loading LLM from {model_path}...")
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() // 2 or 2,
            temperature=0.4,
            verbose=False,
        )

        # Prompt and memory
        self.prompt = self.create_prompt_template()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.chain = self.prompt | self.llm

        logger.info("VoiceChatbot initialized successfully.")

    def create_prompt_template(self):
        """Create a system prompt for natural, concise voice responses."""
        template = """You are a friendly voice assistant named Gunjan.
You help with a wide range of tasks and answer questions accurately.
If you don't know something, say "I don't know, I will update it soon."
Speak naturally like a human. Responses must be concise (max 5 sentences) because this is audio output.
Account for possible speech recognition errors in user input."""
        
        return ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

    def convert_to_wav(self, file_path: str) -> str:
        """Convert any audio file to WAV format for compatibility."""
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.rsplit('.', 1)[0] + "_converted.wav"
        audio.export(wav_path, format="wav")
        logger.debug(f"Converted {file_path} to {wav_path}")
        return wav_path

    def transcribe_audio_file(self, file_path: str) -> str | None:
        """Transcribe a saved audio file using Whisper."""
        try:
            result = self.transcriber(file_path)
            text = result["text"].strip()
            if text:
                logger.info(f"Transcribed: {text}")
                return text
            else:
                logger.warning("Empty transcription result.")
                return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def get_voice_input(self, save_path: str = "recorded_audio.wav") -> str | None:
        """Record audio from microphone and save as WAV."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info("Listening for user input...")
            audio = self.recognizer.listen(source, phrase_time_limit=10)

        try:
            with open(save_path, "wb") as f:
                f.write(audio.get_wav_data())
            logger.info(f"Audio recorded and saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving recorded audio: {e}")
            return None

    def speak_text(self, text: str, speed: float = 1.0, play_audio: bool = True):
        """Convert text to speech using gTTS and play it."""
        if not text.strip():
            return

        clean_text = re.sub(r'[*!#]', '', text)
        logger.info(f"Speaking: {clean_text}")

        tts = gTTS(text=clean_text, lang='en')
        mp3_path = "uploads/result.mp3"
        os.makedirs("uploads", exist_ok=True)
        tts.save(mp3_path)

        # Display in Jupyter if available
        try:
            display(Audio(mp3_path, autoplay=True))
        except:
            pass

        if play_audio:
            audio = AudioSegment.from_mp3(mp3_path)

            # Speed adjustment
            if speed != 1.0:
                altered = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed)
                })
                audio = altered.set_frame_rate(audio.frame_rate)

            play(audio)

    def listen_for_stop(self, timeout_seconds: int = 15) -> bool:
        """Listen for 'stop' command using the audio classifier."""
        sampling_rate = self.classifier.feature_extractor.sampling_rate
        logger.info("Listening for 'stop' command... Say 'STOP' to end conversation.")

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=0.2,
            stream_chunk_s=0.25
        )

        start_time = time.time()
        for prediction in self.classifier(mic):
            if time.time() - start_time > timeout_seconds:
                logger.info("Stop listening timeout reached.")
                return False

            top = prediction[0]
            if top["label"].lower() == "stop" and top["score"] > 0.5:
                logger.info("'STOP' command detected.")
                return True

        return False

    def get_response(self, user_input: str) -> str:
        """Generate LLM response using conversation history."""
        if not user_input:
            return "I didn't catch that. Could you repeat?"

        try:
            response = self.chain.invoke({
                "input": user_input,
                "history": self.memory.chat_memory.messages
            })
            response_text = response.content if hasattr(response, "content") else str(response)
            self.memory.save_context({"input": user_input}, {"output": response_text})

            logger.info(f"LLM Response: {response_text}")
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, something went wrong while generating the response."

    def run(self):
        """Main loop for voice interaction."""
        self.speak_text("Hi, I am Gunjan, your intelligent voice assistant. How can I help you?", speed=1.1)

        max_retries = 3
        while True:
            audio_file = self.get_voice_input()
            if not audio_file or not os.path.exists(audio_file):
                self.speak_text("I couldn't record audio. Please try again.")
                continue

            # Transcribe using Whisper (more accurate than Google API)
            user_text = self.transcribe_audio_file(audio_file)

            if not user_text:
                self.speak_text("Sorry, I didn't understand that. Can you please repeat?")
                continue

            logger.info(f"User query: {user_text}")

            # Get LLM response
            response = self.get_response(user_text)
            self.speak_text(response, speed=1.1)

            # Check if user wants to stop
            if self.listen_for_stop(timeout_seconds=12):
                self.speak_text("Thanks, it was great assisting you. Goodbye!", speed=1.1)
                break


if __name__ == "__main__":
    chatbot = VoiceChatbot(model_path=r'./models/gemma-2-2b-it-Q4_K_M.gguf')
    chatbot.run()