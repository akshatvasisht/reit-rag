"""Lint Python source files for session-tag style comments.

Comments such as "Fix C added this", "Phase 2 cleanup", "S3 task",
"P0-blocker patch", "audit P2 finding", "post-review hardening",
"Round 3 refactor" or "(1a) check this" are workflow artifacts that
decay quickly and pollute code over time. Production code comments
should explain WHY a piece of logic exists, not where in some external
process it originated.

This linter walks the configured roots, tokenizes each ``.py`` file,
and reports comment tokens whose text matches the session-tag pattern.
String literals are skipped automatically because ``tokenize`` yields
only real COMMENT tokens.
"""

from __future__ import annotations

import argparse
import re
import sys
import tokenize
from pathlib import Path

SKIP_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".gitnexus",
    "agentcontext",
    "data",
    "logs",
}

DEFAULT_ROOTS = ("src", "scripts", "tests", "app.py")

PATTERN = re.compile(
    r"^\s*(Fix [A-Z]\b|Phase \d+\b|S\d+\b|P[0-3]-|audit P\d|post-review|Round \d+|\(\d+[a-z]\)\b)",
    re.IGNORECASE,
)


def iter_python_files(root: Path):
    """Yield Python files under ``root`` skipping known noise directories."""
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        yield path


def scan_file(path: Path):
    """Return a list of ``(lineno, comment_text)`` matches for one file."""
    findings = []
    try:
        with path.open("rb") as fh:
            tokens = list(tokenize.tokenize(fh.readline))
    except (SyntaxError, tokenize.TokenizeError, OSError):
        return findings

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        raw = tok.string
        body = raw.lstrip("#")
        if PATTERN.match(body):
            findings.append((tok.start[0], raw.strip()))
    return findings


def resolve_roots(repo_root: Path, paths):
    return [(repo_root / p) if not Path(p).is_absolute() else Path(p) for p in paths]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Flag session-tag comments (Fix X / Phase N / S# / P0- / etc.)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output; only set the exit code.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Override the default scan roots (src scripts tests app.py).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    raw_roots = args.paths if args.paths else list(DEFAULT_ROOTS)
    roots = resolve_roots(repo_root, raw_roots)

    all_findings = []
    seen = set()
    for root in roots:
        for path in iter_python_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            for lineno, text in scan_file(path):
                all_findings.append((path, lineno, text))

    if all_findings and not args.quiet:
        for path, lineno, text in all_findings:
            try:
                display = path.relative_to(Path.cwd())
            except ValueError:
                display = path
            print(f"{display}:{lineno}: {text}")

    return 1 if all_findings else 0


if __name__ == "__main__":
    sys.exit(main())
