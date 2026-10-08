"""Fix one pinned Headless serialization defect, preserving assistant blank lines.

This changes no models, mutation operators, sampling, scoring, or credentials.
The upstream parser removes interior blank lines and corrupts exact SEARCH blocks.
"""
import json


def parse_stdout_preserving_blank_lines(stdout):
    lines = stdout.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError('Headless stdout was empty.')
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError('Headless stdout did not end with usage JSON.') from exc
    usage = payload.get('usage') if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        raise ValueError('Headless usage JSON must contain a top-level usage object.')
    content = '\n'.join(lines[:-1]).strip()
    if not content:
        raise ValueError('Headless assistant content was empty.')
    return content, usage


def install():
    from shinka.llm.providers import headless
    headless._parse_stdout = parse_stdout_preserving_blank_lines
