#!/usr/bin/env python3
"""
tool_calling_chat.py

Interactive chat + tool-calling agent (single-file).

Features:
- Multi-turn chat (keeps conversation history in `messages`).
- LangChain tool-calling loop (executes tools requested by the LLM).
- Tool output chunking: sends ToolMessage content in chunks of max 500 chars.
- Console printing chunked output to max 500 chars per block.

Commands:
- /exit : quit
- /reset: clear conversation
- /url <youtube_url> : summarize a YouTube video
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from typing import Any, Dict, List

import yt_dlp
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

# -----------------------------
# Hygiene: reduce noisy logs
# -----------------------------
warnings.filterwarnings("ignore")

pytube_logger = logging.getLogger("pytube")
pytube_logger.setLevel(logging.ERROR)

ytdlp_logger = logging.getLogger("yt_dlp")
ytdlp_logger.setLevel(logging.ERROR)

# -----------------------------
# Chunking (hard limit 500)
# -----------------------------
MAX_TOOLMSG_LEN = 500      # tool -> LLM (hard requirement)
MAX_PRINT_LEN = 500        # console output chunking (optional, but requested)


def split_into_chunks(text: str, max_len: int) -> list[str]:
    """Split text into chunks where each chunk length <= max_len (tries to split on spaces)."""
    text = str(text or "")
    chunks: list[str] = []
    i = 0

    while i < len(text):
        j = min(i + max_len, len(text))

        # prefer breaking at whitespace
        if j < len(text):
            k = text.rfind(" ", i, j)
            if k != -1 and k > i + 50:  # avoid extremely small chunk
                j = k

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)

        i = j
        while i < len(text) and text[i].isspace():
            i += 1

    return chunks or [""]


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def append_tool_result_chunked(
    messages: list[Any],
    tool_call_id: str,
    tool_result: Any,
    max_len: int = MAX_TOOLMSG_LEN
) -> None:
    """Append tool output as ToolMessage(s), each <= max_len chars."""
    try:
        if isinstance(tool_result, (dict, list, tuple)):
            text = json.dumps(tool_result, ensure_ascii=False, indent=2, default=str)
        else:
            text = str(tool_result)
    except Exception:
        text = str(tool_result)

    chunks = split_into_chunks(text, max_len=max_len)
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        prefix = f"[chunk {idx}/{total}] "
        payload = prefix + chunk

        # hard guarantee
        if len(payload) > max_len:
            payload = payload[:max_len]

        messages.append(ToolMessage(content=payload, tool_call_id=tool_call_id))


def print_chunked(text: str, max_len: int = MAX_PRINT_LEN) -> None:
    for ch in split_into_chunks(text, max_len=max_len):
        print(ch)

# -----------------------------
# Tools (LangChain @tool)
# -----------------------------
@tool
def extract_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from a YouTube URL."""
    pattern = r"(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url or "")
    return match.group(1) if match else "Error: Invalid YouTube URL"

@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """Fetch YouTube transcript text for a video_id in the requested language."""
    try:
        # Try newer style; fallback to classic get_transcript
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id, languages=[language])
            return " ".join([snip.text for snip in transcript.snippets])
        except Exception:
            items = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            return " ".join([x.get("text", "") for x in items if isinstance(x, dict)])
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_youtube(query: str, max_results: int = 5) -> str:
    """Search YouTube using yt-dlp and return a JSON list of results."""
    try:
        n = int(max_results)
        n = max(1, min(n, 25))
    except Exception:
        n = 5

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)

        entries = info.get("entries", []) if isinstance(info, dict) else []
        results: List[Dict[str, Any]] = []

        for e in entries:
            if not isinstance(e, dict):
                continue
            vid = e.get("id")
            title = e.get("title")
            url = e.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
            results.append(
                {
                    "id": vid,
                    "title": title,
                    "url": url,
                    "channel": e.get("channel") or e.get("uploader"),
                    "duration": e.get("duration"),
                }
            )

        return _safe_json(results)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_full_metadata(url: str) -> str:
    """Get full metadata for a YouTube URL using pytube and yt-dlp."""
    out: Dict[str, Any] = {"url": url}

    # pytube metadata
    try:
        yt = YouTube(url)
        out["pytube"] = {
            "video_id": yt.video_id,
            "title": yt.title,
            "author": yt.author,
            "channel_id": yt.channel_id,
            "length_seconds": yt.length,
            "views": yt.views,
            "publish_date": str(yt.publish_date) if yt.publish_date else None,
            "description": yt.description,
            "keywords": yt.keywords,
            "thumbnail_url": yt.thumbnail_url,
        }
    except Exception as e:
        out["pytube_error"] = str(e)

    # yt-dlp metadata
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if isinstance(info, dict):
            out["ytdlp"] = {
                "id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "channel": info.get("channel"),
                "channel_id": info.get("channel_id"),
                "duration": info.get("duration"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "comment_count": info.get("comment_count"),
                "upload_date": info.get("upload_date"),
                "description": info.get("description"),
                "tags": info.get("tags"),
                "webpage_url": info.get("webpage_url"),
            }
    except Exception as e:
        out["ytdlp_error"] = str(e)

    return _safe_json(out)


@tool
def get_thumbnails(url: str) -> str:
    """Return available thumbnail URLs for a YouTube URL."""
    out: Dict[str, Any] = {"url": url, "thumbnails": []}

    # pytube thumbnail
    try:
        yt = YouTube(url)
        if yt.thumbnail_url:
            out["thumbnails"].append({"source": "pytube", "url": yt.thumbnail_url})
    except Exception as e:
        out["pytube_error"] = str(e)

    # yt-dlp thumbnails
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        thumbs = info.get("thumbnails", []) if isinstance(info, dict) else []
        for t in thumbs:
            if isinstance(t, dict) and t.get("url"):
                out["thumbnails"].append(
                    {
                        "source": "yt-dlp",
                        "url": t.get("url"),
                        "width": t.get("width"),
                        "height": t.get("height"),
                        "id": t.get("id"),
                    }
                )
    except Exception as e:
        out["ytdlp_error"] = str(e)

    return _safe_json(out)

# -----------------------------
# Chat agent
# -----------------------------
def build_query_from_url(url: str, language: str) -> str:
    return (
        "Summarize this YouTube video in English.\n"
        f"Video URL: {url}\n"
        f"Preferred transcript language: {language}\n\n"
        "Use tools as needed (extract video id, fetch transcript, metadata, thumbnails). "
        "Return:\n"
        "- 8-12 bullet summary\n"
        "- Key takeaways\n"
        "- If transcript is unavailable, use metadata to infer topic and outline.\n"
    )


def chat_loop(model: str, temperature: float, language: str, max_tool_iterations: int) -> None:
    llm = ChatOllama(model=model, temperature=temperature)

    tools = [extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_thumbnails]
    tool_mapping = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: list[Any] = []

    print("Chat ready. Commands: /url <link>, /reset, /exit")

    while True:
        user_text = input("you> ").strip()

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            break

        if user_text.lower() == "/reset":
            messages = []
            print("(conversation reset)")
            continue

        if user_text.lower().startswith("/url "):
            url = user_text[5:].strip()
            user_text = build_query_from_url(url, language)

        messages.append(HumanMessage(content=user_text))

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        iterations = 0
        while getattr(response, "tool_calls", None) and iterations < max_tool_iterations:
            iterations += 1

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args") or {}
                tool_id = tool_call.get("id")

                if tool_name not in tool_mapping:
                    tool_result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        tool_result = tool_mapping[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Error executing '{tool_name}': {str(e)}"

                # tool -> LLM chunking (<= 500)
                append_tool_result_chunked(messages, tool_id, tool_result, max_len=MAX_TOOLMSG_LEN)

            response = llm_with_tools.invoke(messages)
            messages.append(response)

        print("assistant>")
        print_chunked(response.content or "", max_len=MAX_PRINT_LEN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat tool-calling agent.")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--max_tool_iterations", type=int, default=12)
    args = parser.parse_args()

    chat_loop(
        model=args.model,
        temperature=args.temperature,
        language=args.language,
        max_tool_iterations=args.max_tool_iterations,
    )


if __name__ == "__main__":
    main()
