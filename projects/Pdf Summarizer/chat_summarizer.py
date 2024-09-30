import sys
from  Server import Server

# Run the Flask app
if __name__ == "__main__":
    server = Server()
    try:
        server.run()
    except SystemExit as e:
        print(f"SystemExit occurred: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)