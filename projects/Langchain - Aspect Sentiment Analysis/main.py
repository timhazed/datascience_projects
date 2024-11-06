import argparse
from SentimentAnalyzer import SentimentAnalyzer
import logging

def set_logger():
        logger = logging.getLogger("SentimentAnalyzer")
        logger.setLevel(logging.INFO)

        # Check if the logger already has handlers (to avoid duplicate logs)
        if not logger.handlers:
            handler = logging.StreamHandler()  # Sends log output to the console
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.info("SentimentAnalyzer initialized.")
        return logger
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aspect-Based Sentiment Analysis using LangChain")
    parser.add_argument("--input", type=str, required=True, help="Location of dataset")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run")

    args = parser.parse_args()

    try:
        logger = set_logger()
        analyzer = SentimentAnalyzer(logger, input_path=args.input, iterations=args.iterations)
        analyzer.run()
    except Exception as e:
        logging.error("An error occurred: %s", str(e))


