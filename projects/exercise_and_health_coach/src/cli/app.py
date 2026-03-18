import argparse
import json
import logging
import sys

from langchain_core.globals import set_debug, set_verbose

from src.config.settings import get_settings
from src.orchestrator.coach import ExerciseCoach
from src.state.conversation_state import create_session


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def print_response(response, verbose: bool = False) -> None:
    """Print coach response in a formatted way."""
    print("\n" + "=" * 60)
    print(response.message)
    print("=" * 60)

    if verbose:
        if response.workout_plan:
            print("\n[Workout Plan JSON]")
            print(json.dumps(response.workout_plan.model_dump(), indent=2, default=str))

        if response.recovery_plan:
            print("\n[Recovery Plan JSON]")
            print(json.dumps(response.recovery_plan.model_dump(), indent=2, default=str))

        if response.audit:
            print("\n[Audit Log]")
            print(json.dumps(response.audit.model_dump(), indent=2, default=str))


def print_welcome_intake() -> None:
    """Print the initial safety intake prompt."""
    print("\n" + "-" * 60)
    print("Welcome! Before we get started, I'd like to know a couple of")
    print("things to ensure your safety:")
    print()
    print("- How old are you?")
    print("- Do you have any health conditions, injuries, or concerns")
    print("  I should know about? (If none, just say 'none')")
    print("-" * 60 + "\n")


def run_interactive(coach: ExerciseCoach, verbose: bool = False) -> None:
    """Run interactive CLI session."""
    print("\n" + "=" * 60)
    print("  Exercise & Recovery Coach")
    print("  Type 'quit' or 'exit' to end the session")
    print("  Type 'reset' to start a new session")
    print("  Type 'status' to see current context")
    print("=" * 60 + "\n")

    state = create_session()
    print(f"Session started: {state.session_id[:8]}...")

    # Show initial safety intake prompt
    print_welcome_intake()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\nGoodbye! Stay active! 💪")
                break

            if user_input.lower() == "reset":
                state = create_session()
                print(f"\nNew session started: {state.session_id[:8]}...")
                print_welcome_intake()
                continue

            if user_input.lower() == "status":
                print("\n[Current Context]")
                print(f"  Session: {state.session_id[:8]}...")
                safety_status = "Complete" if state.minimal_intake_complete else "Pending"
                print(f"  Safety intake: {safety_status}")
                print(f"  Full intake: {'Complete' if state.intake_complete else 'Pending'}")
                if state.user_context.biometrics.age:
                    print(f"  Age: {state.user_context.biometrics.age}")
                if state.user_context.fitness_goals:
                    goals = [g.value for g in state.user_context.fitness_goals]
                    print(f"  Goals: {', '.join(goals)}")
                if state.user_context.experience_level:
                    print(f"  Experience: {state.user_context.experience_level}")
                if state.user_context.medical_history.conditions:
                    conditions = ", ".join(state.user_context.medical_history.conditions)
                    print(f"  Conditions: {conditions}")
                print()
                continue

            # Process message
            response, state = coach.process_message(user_input, state)
            print_response(response, verbose)
            print()

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            print(f"\nError: {e}\n")


def run_single(coach: ExerciseCoach, message: str, verbose: bool = False) -> None:
    """Run single message (non-interactive)."""
    state = create_session()
    response, _ = coach.process_message(message, state)
    print_response(response, verbose)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exercise & Recovery Coach CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive mode
  %(prog)s -m "I want to build muscle"  # Single message
  %(prog)s -v                        # Verbose mode with JSON output
        """,
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Single message to process (non-interactive)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed JSON output",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Show detailed debug output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Used for agent tracing and debugging
    if (args.debug):
        set_debug(True)
    
    setup_logging(args.log_level)
    settings = get_settings()

    try:
        coach = ExerciseCoach(settings=settings)
    except Exception as e:
        print(f"Error initializing coach: {e}")
        print("\nMake sure you have set up your .env file with API keys.")
        sys.exit(1)

    # Run
    if args.message:
        run_single(coach, args.message, args.verbose)
    else:
        run_interactive(coach, args.verbose)


if __name__ == "__main__":
    main()
