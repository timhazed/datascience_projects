import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import jsonlines
from typing import List, Dict, Any
import os
from bert_score import BERTScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QdrantQuery")

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        env_path = Path(config.get("env_path")).expanduser()

        # Validate that the .env file exists
        if not env_path or not os.path.isfile(env_path):
            logger.error(f".env file not found at {env_path}")
            raise FileNotFoundError(".env path is invalid or not found in config.yaml")

        logger.info(f"Loading environment variables from: {env_path}")
            
        # Load environment variables from the .env file
        load_dotenv(env_path)

        # Validate required environment variables
        if not os.getenv("QDRANT_API_KEY"):
            raise ValueError("Missing required environment variable: QDRANT_API_KEY")
        
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_qa_pairs(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load question-answer pairs from JSONL file"""
    qa_pairs = []
    try:
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                if "question" in obj and "answer" in obj:
                    qa_pairs.append(obj)
                else:
                    logger.warning(f"Skipping invalid entry: {obj}")
        return qa_pairs
    except Exception as e:
        logger.error(f"Error loading QA pairs: {str(e)}")
        raise

def connect_to_qdrant(config: dict):
    url = config["qdrant_url"]
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(
            url=url,
            api_key=api_key)
    return client

def evaluate_results_with_context(enhanced_results: List[Dict[str, Any]], ground_truth: str) -> Dict[str, Any]:
    """Evaluate search results with context against ground truth using BERTScore"""
    try:
        # Initialize BERTScorer
        scorer = BERTScorer(lang="en")
        
        # Prepare lists for BERTScore while preserving metadata
        candidates = []
        metadata = []
        
        for enhanced_hit in enhanced_results:
            hit = enhanced_hit["main"]
            candidates.append(hit.payload.get("content", ""))
            
            # Extract metadata from the main hit
            hit_metadata = hit.payload.get("metadata", {})
            metadata.append({
                "doc_id": hit_metadata.get("doc_id"),
                "section_title": hit_metadata.get("section_title"),
                "chunk_index": hit_metadata.get("chunk_index"),
                "paragraph_id": hit_metadata.get("paragraph_id"),
                "paragraph_index": hit_metadata.get("paragraph_index"),
                "title": hit_metadata.get("title"),
                "source_url": hit_metadata.get("source_url"),
                "qdrant_score": hit.score
            })
        
        references = [ground_truth] * len(candidates)
        
        # Calculate BERTScore
        P, R, F1 = scorer.score(candidates, references)
        
        # Convert tensors to lists properly
        P_list = [score.item() for score in P]
        R_list = [score.item() for score in R]
        F1_list = [score.item() for score in F1]
        
        # Get the best matching chunk
        best_match_idx = F1.argmax().item()
        best_match_score = F1[best_match_idx].item()
        best_match_content = candidates[best_match_idx]
        best_match_metadata = metadata[best_match_idx]
        
        return {
            "total_chunks": len(enhanced_results),
            "best_match_score": best_match_score,
            "best_match_content": best_match_content[:200] + "..." if len(best_match_content) > 200 else best_match_content,
            "best_match_metadata": best_match_metadata,
            "average_f1": F1.mean().item(),
            "average_precision": P.mean().item(),
            "average_recall": R.mean().item(),
            "all_scores": {
                "precision": P_list,
                "recall": R_list,
                "f1": F1_list,
                "metadata": metadata
            }
        }
    except Exception as e:
        logger.error(f"Error in BERTScore evaluation: {str(e)}")
        return {
            "error": str(e),
            "total_chunks": len(enhanced_results)
        }

def clear_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Clear all points from a Qdrant collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to clear
    """
    try:
        # First check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            logger.warning(f"Collection {collection_name} does not exist")
            return
            
        # Clear all points
        client.clear_payload(
            collection_name=collection_name,
            points_selector=models.Filter(
                must=[]  # Empty filter matches all points
            )
        )
        logger.info(f"Cleared all points from collection {collection_name}")
            
    except Exception as e:
        logger.error(f"Error clearing collection {collection_name}: {str(e)}")
        raise

def points_count(client: QdrantClient, collection_name: str):
    # Simply use the built-in count method
    count_response = client.count(
        collection_name=collection_name
    )
    print("Total points: in collection {}:{}".format(collection_name, count_response.count))

def list_collections(client: QdrantClient):
    # Fetch high-level info about all collections
    collections_info = client.get_collections()

    for coll in collections_info.collections:
        print(f"- Name: {coll.name}")

def main():
    parser = argparse.ArgumentParser(description="Query Qdrant and evaluate against ground truth")
    parser.add_argument("--collection", default="ws1-test", help="Qdrant collection name")
    parser.add_argument("--input", default="data/qa_pairs.jsonl", help="Path to JSONL file with question-answer pairs")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return per query")
    parser.add_argument("--context_window", type=int, default=1, help="Number of adjacent paragraphs to include before and after")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        qdrant_config = config["qdrant"]

        # Initialize Qdrant client
        client = connect_to_qdrant(qdrant_config)

        list_collections(client)
        points_count(client, args.collection)

        # Load QA pairs
        qa_pairs = load_qa_pairs(args.input)

        logger.info(f"Loaded {len(qa_pairs)} question-answer pairs")

        # Uncomment for debugging to clear collection
        #clear_collection(client, args.collection)   

        if (client is None):
            logger.error("Failed to connect to Qdrant")
            raise Exception("Failed to connect to Qdrant")

        # Initialize embedding model
        model = SentenceTransformer(qdrant_config["embedding_model"])

        for i, qa_pair in enumerate(qa_pairs, 1):
            question = qa_pair["question"]
            answer = qa_pair["answer"]

            print(f"\nProcessing question {i}/{len(qa_pairs)}")
            print(f"Question: {question}")
            print(f"Expected answer: {answer}")

            # Generate embedding for the question
            query_embedding = model.encode(question)

            # Search Qdrant
            search_result = client.search(
                collection_name=args.collection,
                query_vector=query_embedding.tolist(),
                limit=args.limit,
                with_payload=True,
                with_vectors=False
            )

            # Get adjacent paragraphs for each result
            enhanced_results = []
            for hit in search_result:
                metadata = hit.payload.get('metadata', {})
                doc_id = metadata.get('doc_id')
                paragraph_id = metadata.get('paragraph_id')
                paragraph_index = metadata.get('paragraph_index')
                
                if doc_id and paragraph_index is not None:
                    # Get adjacent paragraphs
                    adjacent_paragraphs = []
                    for offset in range(-args.context_window, args.context_window + 1):
                        if offset == 0:
                            continue  # Skip the current paragraph as we already have it
                        
                        # Search for adjacent paragraph
                        adjacent_result = client.scroll(
                            collection_name=args.collection,
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="metadata.doc_id",
                                        match=models.MatchValue(value=doc_id)
                                    ),
                                    models.FieldCondition(
                                        key="metadata.paragraph_id",
                                        match=models.MatchValue(value=paragraph_id)
                                    ),
                                    models.FieldCondition(
                                        key="metadata.paragraph_index",
                                        match=models.MatchValue(value=paragraph_index + offset)
                                    )
                                ]
                            ),
                            limit=1,
                            with_payload=True,
                            with_vectors=False
                        )
                        
                        if adjacent_result[0]:  # If we found an adjacent paragraph
                            adjacent_paragraphs.append(adjacent_result[0][0])
                    
                    # Only add result if main content exists
                    if hit.payload.get('content', '').strip():
                        # Combine current hit with adjacent paragraphs
                        enhanced_hit = {
                            "main": hit,
                            "adjacent": adjacent_paragraphs
                        }
                        enhanced_results.append(enhanced_hit)
                else:
                    # Only add result if main content exists
                    if hit.payload.get('content', '').strip():
                        enhanced_results.append({"main": hit, "adjacent": []})

            # Evaluate results with context
            evaluation = evaluate_results_with_context(enhanced_results, answer)

            # Print results
            print(f"\nFound {len(search_result)} results")
            print("\nBERTScore Evaluation:")
            print(f"Best Match Score (F1): {evaluation['best_match_score']:.4f}")
            print(f"Average F1: {evaluation['average_f1']:.4f}")
            print(f"Average Precision: {evaluation['average_precision']:.4f}")
            print(f"Average Recall: {evaluation['average_recall']:.4f}")
            
            print("\nDetailed Results:")
            for j, (enhanced_hit, f1_score) in enumerate(zip(enhanced_results, evaluation['all_scores']['f1']), 1):
                hit = enhanced_hit["main"]
                adjacent = enhanced_hit["adjacent"]
                
                print(f"\nResult {j}:")
                print(f"Qdrant Score: {hit.score:.4f}")
                print(f"BERTScore F1: {f1_score:.4f}")
                
                # Print main content with context
                print("\nMain Content:")
                print(f"{hit.payload.get('content', '')[:200]}...")
                
                if adjacent:
                    print("\nAdjacent Context:")
                    for adj_hit in adjacent:
                        print(f"Adjacent Paragraph: {adj_hit.payload.get('content', '')[:200]}...")
                
                # Print metadata
                metadata = hit.payload.get('metadata', {})
                print("\nMetadata:")
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"Source URL: {metadata.get('source_url', 'N/A')}")
                print(f"Document ID: {metadata.get('doc_id', 'N/A')}")
                print(f"Chunk ID: {metadata.get('chunk_id', 'N/A')}")
                print(f"Section Title: {metadata.get('section_title', 'N/A')}")
                print(f"Paragraph Id: {metadata.get('paragraph_id', 'N/A')}")
                print(f"Paragraph Index: {metadata.get('paragraph_index', 'N/A')}")
                print("-" * 80)

    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise

if __name__ == "__main__":
    main()
