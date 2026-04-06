from src.utils.constants import MAX_HISTORY_TURNS


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Retain the most recent max_turns complete turns (user + assistant pairs).
    One turn = 2 messages. Older turns dropped FIFO.
    Returns a new list — does not mutate state in-place.
    """
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history
