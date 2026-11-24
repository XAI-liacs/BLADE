# Adapted for BLADE (iohblade) from ShinkaEvolve 0.0.8
# (https://github.com/SakanaAI/ShinkaEvolve, shinka/edit), Copyright Sakana AI,
# licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changes: the whole program is
# editable (no EVOLVE-BLOCK markers), and the matching is simplified to exact
# and indentation-tolerant matches.
"""Parse LLM replies and apply SEARCH/REPLACE diffs."""

from __future__ import annotations

import re
import textwrap

PATCH_PATTERN = re.compile(
    r"<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}\s*REPLACE",
    re.DOTALL,
)
_NAME = re.compile(r"<NAME>\s*(.*?)\s*</NAME>", re.DOTALL)
_DESCRIPTION = re.compile(r"<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>", re.DOTALL)
_CODE_BLOCK = re.compile(r"<CODE>(.*?)</CODE>", re.DOTALL)
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


class PatchError(ValueError):
    """The LLM reply could not be turned into a program."""


def extract_tag(reply: str, tag: str) -> str:
    """The text between ``<TAG>`` and ``</TAG>`` (empty if missing)."""
    pattern = {"NAME": _NAME, "DESCRIPTION": _DESCRIPTION}[tag]
    match = pattern.search(reply)
    return match.group(1).strip() if match else ""


def extract_code(reply: str) -> str:
    """The program of a full rewrite: the fenced code inside ``<CODE>``,
    or else the last fenced code block of the reply."""
    block = _CODE_BLOCK.search(reply)
    text = block.group(1) if block else reply
    fences = _FENCE.findall(text)
    if fences:
        code = fences[0] if block else max(fences, key=len)
    elif block:
        code = text
    else:
        raise PatchError("The reply contains no <CODE> block or fenced code.")
    code = code.strip("\n")
    if not code.strip():
        raise PatchError("The reply contains an empty program.")
    return code + "\n"


def _strip_trailing(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_indented(search: str, code: str) -> tuple[int, int, str] | None:
    """Find ``search`` in ``code`` ignoring a uniform indentation offset.

    Returns (start, end, indentation of the match) on the line level.
    """
    search_lines = textwrap.dedent(search).split("\n")
    lines = code.split("\n")
    n = len(search_lines)
    for i in range(len(lines) - n + 1):
        window = lines[i : i + n]
        first = next((line for line in window if line.strip()), None)
        if first is None:
            continue
        indent = _indent_of(first)
        candidate = [line[len(indent) :] if line.strip() else "" for line in window]
        target = [line.rstrip() for line in search_lines]
        if [line.rstrip() for line in candidate] == target and all(
            not line.strip() or line.startswith(indent) for line in window
        ):
            return i, i + n, indent
    return None


def apply_diff(code: str, reply: str) -> tuple[str, int]:
    """Apply the SEARCH/REPLACE blocks of ``reply`` to ``code`` in order.

    A SEARCH text must occur in the program, exactly or with a different
    indentation (then the replacement is re-indented the same way).

    Returns:
        tuple[str, int]: The new program and the number of blocks applied.

    Raises:
        PatchError: When the reply has no blocks, a SEARCH text is not
            found, or the result equals the original program.
    """
    blocks = PATCH_PATTERN.findall(reply)
    if not blocks:
        raise PatchError("No SEARCH/REPLACE blocks found in the reply.")
    new = code
    for i, (search, replace) in enumerate(blocks, start=1):
        if search in new:
            new = new.replace(search, replace, 1)
            continue
        stripped = _strip_trailing(search)
        if stripped and stripped in _strip_trailing(new):
            new = _strip_trailing(new).replace(stripped, _strip_trailing(replace), 1)
            continue
        found = _find_indented(search, new)
        if found is None:
            preview = search.strip().splitlines()[0] if search.strip() else ""
            raise PatchError(
                f"SEARCH block {i} was not found in the program "
                f"(first line: {preview!r}). Copy SEARCH text verbatim."
            )
        start, end, indent = found
        lines = new.split("\n")
        replacement = [
            indent + line if line.strip() else line
            for line in textwrap.dedent(replace).split("\n")
        ]
        new = "\n".join(lines[:start] + replacement + lines[end:])
    if new == code:
        raise PatchError("The edit does not change the program.")
    return new, len(blocks)
