import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config.settings import get_settings
from src.orchestrator.coach import ExerciseCoach
from src.orchestrator.state_router import WorkflowType
from src.state.conversation_state import create_session

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_questions(input_path: Path) -> list[dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return questions


def run_validation(
    input_path: Path,
    output_path: Path,
    verbose: bool = False,
) -> dict:
    """
    Run validation against a set of questions.

    Args:
        input_path: Path to input JSONL file with questions.
        output_path: Path to output JSONL file for results.
        verbose: If True, print progress to stdout.

    Returns:
        Summary statistics dict.
    """
    # Load questions
    questions = load_questions(input_path)
    if not questions:
        print(f"No valid questions found in {input_path}")
        return {"total": 0, "processed": 0, "errors": 0}

    if verbose:
        print(f"Loaded {len(questions)} questions from {input_path}")

    # Initialize coach
    settings = get_settings()
    coach = ExerciseCoach(settings=settings)

    # Process each question
    results = []
    errors = 0
    intent_hits = 0
    redflag_misses = 0
    false_blocks = 0

    for i, item in enumerate(questions, 1):
        question = item.get("question", "")
        expected_intent = item.get("expected_intent", "")

        if verbose:
            print(f"\n[{i}/{len(questions)}] Processing: {question[:60]}...")

        try:
            # Create fresh session for each question
            state = create_session()

            # Pre-route to capture predicted intent/workflow for scoring
            preliminary_routing = coach.router.route(question, state)

            # Process through coach
            response, state = coach.process_message(question, state)

            # Build result
            result = {
                "question": question,
                "expected_intent": expected_intent,
                "predicted_intent": preliminary_routing.intent.intent.value,
                "workflow": preliminary_routing.workflow.value,
                "predicted_response": response.message,
                "is_rejected": response.is_rejected,
                "needs_more_info": response.needs_more_info,
                "has_workout_plan": response.workout_plan is not None,
                "has_recovery_plan": response.recovery_plan is not None,
            }

            # Add audit info if available
            if response.audit:
                result["audit_status"] = response.audit.status.value

            # Scoring
            if expected_intent == "red_flag_check":
                is_blocked = preliminary_routing.workflow == WorkflowType.SAFETY_BLOCK
                if response.is_rejected or is_blocked:
                    intent_hits += 1
                else:
                    redflag_misses += 1
            else:
                if expected_intent == preliminary_routing.intent.intent.value:
                    intent_hits += 1
                elif preliminary_routing.workflow == WorkflowType.SAFETY_BLOCK:
                    false_blocks += 1

            results.append(result)

            if verbose:
                status = "REJECTED" if response.is_rejected else "OK"
                print(f"    Status: {status}")

        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            errors += 1
            results.append({
                "question": question,
                "expected_intent": expected_intent,
                "predicted_response": f"ERROR: {str(e)}",
                "error": True,
            })

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\nResults written to {output_path}")

    return {
        "total": len(questions),
        "processed": len(results),
        "errors": errors,
        "intent_hits": intent_hits,
        "intent_accuracy": round(intent_hits / len(questions), 3) if questions else 0.0,
        "redflag_misses": redflag_misses,
        "false_blocks": false_blocks,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run validation against a set of questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/Validation_Questions.jsonl
  %(prog)s data/Validation_Questions.jsonl -o results/
  %(prog)s data/Validation_Questions.jsonl -v
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input JSONL file with validation questions",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for output file (default: data/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress to stdout",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    input_filename_prefix = args.input.stem

    # Create output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{input_filename_prefix}_Run_{timestamp}.jsonl"
    output_path = args.output_dir / output_filename

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Running validation...")
    print(f"  Input:  {args.input}")
    print(f"  Output: {output_path}")

    try:
        stats = run_validation(args.input, output_path, verbose=args.verbose)
        print(f"\nDone! Processed {stats['processed']}/{stats['total']} questions")
        if stats["errors"] > 0:
            print(f"  Errors: {stats['errors']}")
        print(f"  Intent accuracy: {stats.get('intent_accuracy', 0):.3f}")
        print(f"  Red-flag misses: {stats.get('redflag_misses', 0)}")
        print(f"  False safety blocks: {stats.get('false_blocks', 0)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
