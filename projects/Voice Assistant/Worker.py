import openai
import os
from gtts import gTTS
from dotenv import load_dotenv, find_dotenv
import io
from flask import jsonify
from gtts.lang import tts_langs
import whisper

class Worker:
    def __init__(self):
        self.client = None
        self.model = None
        self.load_env()
        self.initialize_openai()
        self.initialize_whisper()

    # Load environment variables from a .env file
    def load_env(self):
        _ = load_dotenv(find_dotenv())

    def get_openai_api_key(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        return openai_api_key

    def initialize_openai(self):
        openai_api_key = self.get_openai_api_key()
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

    def initialize_whisper(self):
        self.model = whisper.load_model("base")

    # Transcribe speech to text using Whisper model
    def speech_to_text(self, temp_audio_file_path):
        response = None
        status_code = 200

        try:
            # Transcribe the temporary audio file
            result = self.model.transcribe(temp_audio_file_path)
            transcript = result['text'].strip()

            if not transcript:
                status_code = 400
                raise ValueError("No transcribable speech detected!", status_code)

            # Return the transcribed text
            return transcript

        except Exception as e:
            # Handle any unexpected exceptions and include the status code in the message
            response = {"error": f"An unexpected error occurred: {str(e)}", "status_code": status_code}

        return jsonify(response), status_code

    # Convert text to speech using gTTS and return the audio
    def text_to_speech(self, text, lang="en"):
        # Get the dictionary of supported languages
        supported_langs = tts_langs()

        # Ensure lang is a valid code, default to "en" if not
        if lang not in supported_langs:
            print(f"Warning: Unsupported language '{lang}'. Defaulting to English.")
            lang = "en"

        # Create gTTS object
        tts = gTTS(text=text, lang=lang)

        # Use in-memory file object to avoid filesystem IO
        with io.BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            audio_data = f.getvalue()

        return audio_data

    # Process user message with OpenAI
    def openai_process_message(self, user_message):
        response_text = None
        prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
        try:
            openai_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=4000
            )
            print("OpenAI response:", openai_response)
            response_text = openai_response.choices[0].message.content

        except Exception as e:
            response_text = f"Error during OpenAI request: {e}"

        return response_text
