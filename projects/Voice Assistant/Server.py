import base64
from flask import Flask, render_template, request, jsonify
from Worker import Worker
from flask_cors import CORS
import os
import tempfile

class Server:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        self.worker = Worker()
        self.register_routes()

    def register_routes(self):
        # Register Flask routes
        self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
        self.app.add_url_rule('/speech-to-text', view_func=self.speech_to_text_route, methods=['POST'])
        self.app.add_url_rule('/process-message', view_func=self.process_message_route, methods=['POST'])

    def index(self):
        return render_template('index.html')

    def speech_to_text_route(self):
        response = None
        status_code = 200

        try:
            if 'audio_data' not in request.files:
                status_code = 400
                raise ValueError("No audio file uploaded!", status_code)

            audio_file = request.files['audio_data']
            if audio_file.filename == '':
                status_code = 400
                raise ValueError("Empty audio file!", status_code)

            # Use a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                audio_file.save(temp_audio_file_path)
                text = self.worker.speech_to_text(temp_audio_file_path)
                response = {'text': text}

        except Exception as e:
            # Handle any unexpected exceptions and include the status code in the message
            response = {"error": f"An unexpected error occurred: {str(e)}", "status_code": status_code}

        return jsonify(response), status_code

    def process_message_route(self):
        response = None
        status_code = 200

        try:
            data = request.get_json()
            if not data:
                status_code = 400
                raise ValueError("No JSON data provided", status_code)

            user_message = data.get('userMessage')

            if not user_message or not user_message.strip():
                status_code = 400
                raise ValueError("User message is empty", status_code)

            voice = data.get('voice', 'en-US')  # Default to en-US if not provided

            openai_response_text = self.worker.openai_process_message(user_message)
            openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])

            openai_response_speech = self.worker.text_to_speech(openai_response_text, voice)
            openai_response_speech = base64.b64encode(openai_response_speech).decode('utf-8')

            response = {
                "openaiResponseText": openai_response_text,
                "openaiResponseSpeech": openai_response_speech
            }

        except Exception as e:
            # Handle any unexpected exceptions and include the status code in the message
            response = {"error": f"An unexpected error occurred: {str(e)}", "status_code": status_code}

        return jsonify(response), status_code

    def run(self):
        self.app.run(port=8000, host='0.0.0.0')
