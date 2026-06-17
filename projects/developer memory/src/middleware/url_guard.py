"""Domain allowlist validator for repository URLs.

Spec §6.3a — shell-metacharacter rejection alone is insufficient against git subcommand
injection. A syntactically clean URL can still encode a malicious host. Strict allowlisting
of known-safe git hosting domains is the required control.
"""

from urllib.parse import urlparse

# Extend via settings.py (ALLOWED_HOSTS env override) for self-hosted GitLab instances.
ALLOWED_HOSTS: frozenset[str] = frozenset({"github.com", "gitlab.com", "bitbucket.org"})


def validate_repo_url(url: str) -> str:
    """Allowlist-based repo URL validator. Rejects anything not on known-safe hosts.

    Rules enforced:
      - Hostname must be in ALLOWED_HOSTS (after stripping any www. prefix)
      - URL path must not contain ".." (path traversal)
      - URL must not contain null bytes (encoding attack)

    Args:
        url: The raw repo URL provided by the MCP client.

    Returns:
        The original url string if valid.

    Raises:
        ValueError: With a safe, non-leaking message on any violation.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    # Strip www. prefix for normalisation — www.github.com == github.com
    host = host.removeprefix("www.")

    if host not in ALLOWED_HOSTS:
        raise ValueError(
            f"repo_url host '{host}' is not in the allowed list: {sorted(ALLOWED_HOSTS)}"
        )

    # Reject path traversal and null bytes — these survive URL parsing unchanged
    if ".." in parsed.path or "\x00" in url:
        raise ValueError("repo_url contains illegal path characters")

    return url
