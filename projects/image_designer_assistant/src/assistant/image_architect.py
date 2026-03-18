import os
import re
import tempfile
from typing import Any

import requests
from uuid import uuid4
from langchain.agents import create_agent
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
from PIL import Image
from io import BytesIO
import gradio as gr

# Pre-compiled regexes (avoids recompilation on every request)
_RE_IMAGE_MARKDOWN = re.compile(r"!\[.*?\]\([^)]+\)")
_RE_RAW_URL = re.compile(r"https?://[^\s\)\]\>]+")
_RE_REDUNDANT_PHRASES = re.compile(
    r"Feel free to download[^.]*\.?|You can (?:view|download) (?:it|the image)[^.]*\.?",
    re.I,
)
_RE_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_RE_MULTIPLE_SPACES = re.compile(r" {2,}")

SYSTEM_PROMPT = """You are a Master Visual Architect.
Use the chat history to understand the evolution of the design.
1. When the user asks for an image, design, banner, or any visual artwork, you MUST use the DALL-E tool to generate it. Never just describe what you would create—actually call the tool.
2. If refining a previous design, incorporate those details into a NEW DALL-E tool call.
3. Be extremely descriptive in your prompts to DALL-E.
4. Only skip the tool for pure conversation (e.g., "hello", "thanks", or questions about the process).
5. If DALL-E refuses (e.g., copyrighted characters like The Minions, Mickey Mouse), politely explain that you cannot generate that specific content due to copyright restrictions, and suggest alternatives (e.g., "yellow cartoon characters in overalls" instead of "The Minions")."""


class ImageArchitectAgent:
    def __init__(self) -> None:
        api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="hd")
        self.image_tool = OpenAIDALLEImageGenerationTool(api_wrapper=api_wrapper)
        self.agent = create_agent(
            model="openai:gpt-3.5-turbo",
            tools=[self.image_tool],
            system_prompt=SYSTEM_PROMPT,
        )

    def _validate_and_save_image(self, img_data: bytes) -> str:
        """Validate image data using PIL and save as PNG to temporary directory.
        
        Returns:
            str: Filepath to the saved PNG image
        
        Raises:
            ValueError: If the image data is invalid or corrupted
        """
        # Validate image by attempting to open with PIL
        try:
            img = Image.open(BytesIO(img_data))
            img.verify()  # Verify image integrity
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image data: {e}")
        
        # Re-open image (verify() closes the file)
        img = Image.open(BytesIO(img_data))
        
        # Save as PNG to temporary directory (converts from WebP if needed)
        filename = f"design_{uuid4().hex[:8]}.png"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        img.save(filepath, format="PNG")
        
        return filepath

    def _clean_response(self, response_text: str) -> str:
        text = (response_text or "").replace("\\n", "\n").strip()
        text = _RE_IMAGE_MARKDOWN.sub("", text)
        text = _RE_RAW_URL.sub("", text)
        text = _RE_REDUNDANT_PHRASES.sub("", text)
        text = _RE_EXCESS_NEWLINES.sub("\n\n", text)
        result = _RE_MULTIPLE_SPACES.sub(" ", text).strip()
        return result

    def generate_image(self, message: str, history: list[dict[str, Any]]) -> gr.ChatMessage | str:
        """
        Gradio 5 passes 'history' as a list of MessageDict: [{"role": "user"|"assistant", "content": "..."}, ...]
        We convert this into LangChain message format for the agent.
        
        Returns:
            gr.ChatMessage with text and PNG image filepath, or plain string for text-only responses
        """
        # Build messages from history + new user input
        messages = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("text", content.get("path", str(content)))
            messages.append({"role": role, "content": str(content) if content else ""})
        messages.append({"role": "user", "content": message})

        # Invoke the agent
        result = self.agent.invoke({"messages": messages})
        response_messages = result["messages"]
        response_text = response_messages[-1].content if response_messages else ""

        # Extract the image URL from the current turn's DALL-E tool output.
        # response_messages contains the full conversation (prior turns + this turn).
        # We must only use URLs from ToolMessages after the user's latest prompt,
        # otherwise we'd show a stale/expired image when DALL-E refuses (e.g. copyrighted characters).

        # Find the index of the last HumanMessage (the user's most recent prompt).
        # Search backwards so we get the latest user message in a multi-turn exchange.
        last_user_idx = next(
            (i for i in range(len(response_messages) - 1, -1, -1) if isinstance(response_messages[i], HumanMessage)),
            -1,
        )

        # Scan messages after last_user_idx for ToolMessages containing a DALL-E URL.
        # ToolMessages hold tool outputs; DALL-E returns the image URL in its ToolMessage content.
        url = None
        for i, msg in enumerate(response_messages):
            if i <= last_user_idx or not isinstance(msg, ToolMessage):
                continue

            content = getattr(msg, "content", None) or ""
            if isinstance(content, str) and "https://" in content:
                # Parse the URL: content may be "https://host/path?params" or have extra text.
                # Take the last "https://" segment (handles edge cases), then first token, strip punctuation.
                raw = content.split("https://")[-1].split()[0].strip("()[].,;")
                url = "https://" + raw

        if url:
            img_data = requests.get(url).content
            
            # Validate and save image as PNG using PIL
            filepath = self._validate_and_save_image(img_data)

            # Clean agent text for display
            text = self._clean_response(response_text)

            # Return ChatMessage with text and PNG filepath for Gradio to display
            return gr.ChatMessage(
                content=[text, gr.Image(value=filepath)] if text else [gr.Image(value=filepath)]
            )

        else:
            return (response_text or "").replace("\\n", "\n").strip()