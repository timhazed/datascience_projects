from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from Worker import Worker  
import logging

class Server:
    def __init__(self):
        """Initialize the Flask app, CORS, and worker."""
        self.app = Flask(__name__)
        self.cors = CORS(self.app, resources={r"/*": {"origins": "*"}})
        self.worker = Worker()  # Instantiate your worker
        self.configure_app()
        self.add_routes()

    def configure_app(self):
        """Configure the app's settings, logging, and other middleware."""
        self.app.logger.setLevel(logging.ERROR)

    def index(self):
        return render_template('index.html')
    
    def add_routes(self):
        """Define all routes for the Flask app."""

        # Define routes by binding them to methods
        self.app.add_url_rule('/', view_func=self.index, methods=['GET'])
        self.app.route('/process-message', methods=['POST'])(self.process_message)
        self.app.route('/process-document', methods=['POST'])(self.process_document)

    
    def process_message(self):
        """Process user messages."""
        user_message = request.json.get('userMessage')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        bot_response = self.worker.process_prompt(user_message)
        return jsonify({"botResponse": bot_response}), 200

    def process_document(self):
        """Process user documents."""
        if 'file' not in request.files:
            return jsonify({
                "botResponse": "File not uploaded correctly, please try again."
            }), 400
        
        file = request.files['file']
        file_path = file.filename
        file.save(file_path)  # Save the file
        self.worker.process_document(file_path)
        
        return jsonify({
            "botResponse": "Document processed successfully."
        }), 200

    def run(self, host='127.0.0.1', port=8000, debug=False):
        """Run the Flask app."""
        self.app.run(ssl_context=(self.worker.cert_path, self.worker.key_path), port=port, host=host, debug=debug)

