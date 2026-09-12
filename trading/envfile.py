"""Read and write single values in a .env file, without disturbing the rest.

Hand-editing .env is where setup goes wrong: a misspelled name is silently
ignored, placeholder brackets get left in, and quotes or stray spaces end up
inside the value. These helpers exist so scripts can set a value correctly
and the operator never has to.
"""

from __future__ import annotations

import re
from pathlib import Path

EXAMPLE_NAME = ".env.example"


def read_value(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, re.M)
    return match.group(1).strip() if match else ""


def clean(value: str) -> str:
    """Strip what people accidentally paste around a value.

    Placeholder brackets from documentation, surrounding quotes, and
    whitespace are all removed — they are never part of a real credential.
    """
    value = value.strip()
    if len(value) > 1 and value[0] == "<" and value[-1] == ">":
        value = value[1:-1].strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def set_value(env_path: Path, name: str, value: str) -> None:
    """Set one variable in .env, creating the file from .env.example if needed."""
    if env_path.exists():
        text = env_path.read_text()
    else:
        example = env_path.with_name(EXAMPLE_NAME)
        text = example.read_text() if example.exists() else ""

    if re.search(rf"^{re.escape(name)}=.*$", text, re.M):
        text = re.sub(
            rf"^{re.escape(name)}=.*$", f"{name}={value}", text, count=1, flags=re.M
        )
    else:
        text = (text.rstrip("\n") + f"\n{name}={value}\n").lstrip("\n")

    env_path.write_text(text)
