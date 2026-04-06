import uuid

import click
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from src.data.response import SupervisorResponse
from src.data.user_query import UserQuery
from src.graph.builder import build_graph

console = Console()
err_console = Console(stderr=True)

# Compiled graph — singleton built once at CLI startup.
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _render_response(response: SupervisorResponse) -> None:
    """Print SupervisorResponse to stdout using Rich formatting."""
    if response.fallback_used:
        console.print(
            "[yellow]⚠ One or more news sources returned no results.[/yellow]"
        )

    for i, section in enumerate(response.sections):
        if i > 0:
            console.print(Rule())
        console.print(Text(f"\n{section.query}", style="bold"))

        if not section.articles:
            console.print("  [dim]No results found for this query.[/dim]")
            continue

        for article in section.articles:
            console.print(
                f"\n  [bold cyan][link={article.url}]{article.title}[/link][/bold cyan]"
            )
            if article.summary:
                console.print(f"  {article.summary}")
            date_str = article.published_at.strftime("%b %d, %H:%M UTC")
            console.print(
                f"  [dim]{article.source_name} · {date_str} · {article.provider}[/dim]"
            )

    if response.sources_used:
        console.print(f"\n[dim]Sources: {', '.join(response.sources_used)}[/dim]")


@click.command()
@click.argument("prompt")
@click.option(
    "--session-id",
    "-s",
    default=None,
    help="Session ID for conversation context. Auto-generated if not provided.",
)
def cli(prompt: str, session_id: str | None) -> None:
    """Query NewsGenie and print results."""
    sid = session_id or str(uuid.uuid4())

    try:
        user_query = UserQuery(text=prompt, session_id=sid)
    except Exception as exc:
        err_console.print(f"[red]Invalid query:[/red] {exc}")
        raise click.exceptions.Exit(code=1) from exc

    from src.data.state import AgentState

    state = AgentState(query=user_query)

    try:
        with console.status("[bold green]Fetching news…[/bold green]", spinner="dots"):
            output = _get_graph().invoke(state)
    except Exception as exc:
        err_console.print(
            f"[red]Error:[/red] I encountered an issue processing your request. {exc}"
        )
        raise click.exceptions.Exit(code=1) from exc

    response: SupervisorResponse = output["final_response"]
    _render_response(response)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
