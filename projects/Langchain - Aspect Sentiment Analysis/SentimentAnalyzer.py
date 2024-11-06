import os
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from langchain_openai import ChatOpenAI
from ReviewAnalysis import ReviewAnalysis
from dotenv import load_dotenv

class SentimentAnalyzer:
    def __init__(self, logger, input_path, iterations=5, batch_size=10):
        self.input_path = input_path
        self.iterations = iterations
        self.batch_size = batch_size
        self.logger = logger
        self.aspects = ['Operational Efficiency', 'Customer Satisfaction', 'Durable Packaging', 'On-time Delivery']
        self.load_data()
        self.get_llm()
        self.structured_llm = self.llm.with_structured_output(ReviewAnalysis)
        
        # Zero shot prompt
        self.prompt_template = """
    Analyze the following review for sentiment and select ONE the following aspects that is the main theme of the review:
    {aspects_text}

    Provide:
    - Aspect: aspect that represents the main theme of the review.
    - Sentiment: either Positive or Negative
    - Polarity: a score between 0 and 1, where 0 is extremely negative, 1 is extremely positive, and values close to 0.5 are neutral.

    Respond in the following JSON format:
    {{
        "review_id": {review_id},
        "aspect": "<Aspect>",
        "sentiment": "<Positive or Negative>",
        "polarity": <float>
    }}

    Review: "{review_text}"
    """

    def load_data(self):
        """
        Load the input dataset from a CSV file and validate its structure.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is empty or does not contain required columns.
        """

        # Load dataset
        try:
            self.df = pd.read_csv(self.input_path)
            if "id" not in self.df.columns or "review" not in self.df.columns or "sentiment" not in self.df.columns:
                raise ValueError("Input dataset must contain 'id', 'review', and 'sentiment' columns.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {self.input_path}.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Dataset file at {self.input_path} is empty or corrupted.")

    def get_llm(self, model="gpt-4o-mini"):
        """
        Initialize the language model (LLM) with an API key loaded from environment variables.
        
        Args:
            model (str): Model name for the language model. Default is 'gpt-4o-mini'.
        
        Raises:
            EnvironmentError: If the API key is not found in the environment.
        """

        dotenv_path = '../../python/.env' 
        _ = load_dotenv(dotenv_path=dotenv_path)
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is missing. Please set it in your environment or .env file.")
        
        # Initialize the language model
        self.llm = ChatOpenAI(model=model, api_key=api_key)

    def analyze_batch(self, batch):
        """
        Analyze a batch of reviews using the LLM to predict sentiment for each specified aspect.
        
        Args:
            batch (pd.DataFrame): A batch of reviews to analyze.
        
        Returns:
            list: A list of dictionaries with structured outputs containing review_id, aspect, sentiment, and polarity.
        """

        # Generate the list of aspects for the prompt dynamically
        aspects_text = "\n".join(f"- {aspect}" for aspect in self.aspects)

        # Create batch prompts by injecting the aspects into the prompt template
        batch_prompts = [
            self.prompt_template.format(
                review_id=row['id'],
                review_text=row['review'],
                aspects_text=aspects_text
            )
            for _, row in batch.iterrows()
        ]

        batch_results = [self.structured_llm.invoke(prompt) for prompt in batch_prompts]
        return [result.dict() for result in batch_results]

    def calculate_polarity_metrics(self, iteration_df):
        """
        Calculate the average polarity across all aspects and by specific aspect.
        
        Args:
            iteration_df (pd.DataFrame): DataFrame containing the polarity and aspect data for each review in the iteration.
            
        Returns:
            tuple: A tuple containing the overall average polarity and a dictionary of average polarity by aspect.
        """

        avg_polarity = iteration_df['polarity'].mean()
        aspect_polarities = {aspect: iteration_df[iteration_df['aspect'] == aspect]['polarity'].mean()
                            for aspect in self.aspects}
        return avg_polarity, aspect_polarities

    def run_iterations(self):
        """
        Run multiple iterations of sentiment analysis and compute metrics for each iteration.
        
        Returns:
            dict: A dictionary containing Micro F1 scores, average polarity per iteration, and average polarity by aspect.
        """

        f1_scores = []
        avg_polarity_scores = []
        aspect_polarity_scores = {aspect: [] for aspect in self.aspects}

        # Compute ground truth sentiments once, sorted by review_id
        true_sentiments = self.df.sort_values("id")["sentiment"].tolist()
        
        # Process each iteration
        for iteration in range(self.iterations):
            self.logger.info("Starting iteration {}".format(iteration))

            run_results = []
            
            # Iterate over the entire dataset, processing in batches
            for i in range(0, len(self.df), self.batch_size):
                batch = self.df.iloc[i : i + self.batch_size]
                predictions = self.analyze_batch(batch)
                run_results.extend(predictions)
            
            # Align predictions with ground truth by sorting run_results by review_id
            pred_sentiments = [pred["sentiment"] for pred in sorted(run_results, key=lambda x: x["review_id"])]
            
            # Calculate Micro F1 Score for this run
            f1 = f1_score(true_sentiments, pred_sentiments, average='micro')
            f1_scores.append(f1)
            
            # Convert run_results to a DataFrame for storage and calculations
            iteration_df = pd.DataFrame(run_results)
            iteration_df = iteration_df[["review_id", "sentiment", "polarity", "aspect"]]

            # Calculate overall average polarity for this iteration and per aspect polarity
            avg_polarity, aspect_avg_polarity = self.calculate_polarity_metrics(iteration_df)
            avg_polarity_scores.append(avg_polarity)
            
            # Store average polarity by aspect
            for aspect, polarity in aspect_avg_polarity.items():
                aspect_polarity_scores[aspect].append(polarity)

            self.logger.info("Completed iteration {}".format(iteration))
        # Store the results in a dictionary for easy access
        results = {
            "f1_scores": f1_scores,
            "Avg Polarity": avg_polarity_scores,
            "Aspect Polarities": aspect_polarity_scores
        }

        return results
 
    def plot_metrics(self, results):
        """
        Plot metrics from the sentiment analysis, including F1 scores and polarity averages over iterations.
        
        Args:
            results (dict): Dictionary containing F1 scores, overall polarity, and per-aspect polarity.
        """

        # Create a 2x2 grid for subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Ensure x-axis uses integer values for iterations
        integer_locator = MaxNLocator(integer=True)

        # Plot Micro F1 Scores over iterations (top-left)
        axs[0, 0].plot(range(1, len(results["f1_scores"]) + 1), results["f1_scores"], marker='o')
        axs[0, 0].set_title("Micro F1 Score over Iterations")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Micro F1 Score")
        axs[0, 0].grid(True)
        axs[0, 0].xaxis.set_major_locator(integer_locator)

        # Plot Overall Average Polarity over iterations (top-right)
        axs[0, 1].plot(range(1, len(results["Avg Polarity"]) + 1), results["Avg Polarity"], marker='o')
        axs[0, 1].set_title("Overall Average Polarity over Iterations")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Average Polarity")
        axs[0, 1].grid(True)
        axs[0, 1].xaxis.set_major_locator(integer_locator)

        # Plot Average Polarity by Aspect over iterations (bottom-left)
        for aspect, polarities in results["Aspect Polarities"].items():
            axs[1, 0].plot(range(1, len(polarities) + 1), polarities, label=aspect, marker='o')
        axs[1, 0].set_title("Average Polarity by Aspect over Iterations")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Average Polarity")
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].xaxis.set_major_locator(integer_locator)

        # Hide the bottom-right subplot (optional if only using 3 plots)
        axs[1, 1].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show(block=True)


    def run(self):
        metrics = self.run_iterations()
        self.plot_metrics(metrics)
