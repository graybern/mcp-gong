"""Merges raw Gong transcript segments with parties metadata into
speaker-labeled, chronologically ordered plain text.

Kept dependency-free (no `mcp` import) so it can be unit tested in isolation
from the MCP tool-registration layer.
"""

from __future__ import annotations

from typing import Any, Optional

ROLE_BY_AFFILIATION = {
    "Internal": "Internal",
    "External": "Customer",
}
UNKNOWN_ROLE = "Unknown"
UNKNOWN_NAME = "Unknown Speaker"


def resolve_role(affiliation: Optional[str]) -> str:
    """Map a Gong party's `affiliation` field to the Customer/Internal label
    used throughout tool output. Anything Gong doesn't classify as Internal
    or External (including missing/unrecognized values) is labeled Unknown
    rather than guessed at."""
    return ROLE_BY_AFFILIATION.get(affiliation, UNKNOWN_ROLE)


def format_transcript(
    transcript: list[dict[str, Any]], parties: list[dict[str, Any]]
) -> str:
    """Return the transcript as plain text, one line per speaker turn:
    `[HH:MM:SS] Role (Name): utterance`, ordered by when each turn started.

    `transcript` is the raw `callTranscripts[].transcript` monologue list from
    Gong (each entry has a `speakerId` and a `sentences` list with `start`
    offsets in milliseconds); `parties` is the call's parties metadata (each
    entry has `speakerId`, `name`, `affiliation`).
    """
    speakers = _index_parties(parties)
    turns = []

    for monologue in transcript:
        sentences = monologue.get("sentences") or []
        text = " ".join(
            sentence.get("text", "").strip() for sentence in sentences if sentence.get("text")
        )
        if not text:
            continue

        start_ms = sentences[0].get("start") or 0
        role, name = speakers.get(monologue.get("speakerId"), (UNKNOWN_ROLE, UNKNOWN_NAME))
        turns.append((start_ms, role, name, text))

    turns.sort(key=lambda turn: turn[0])

    return "\n".join(
        f"[{_format_timestamp(start_ms)}] {role} ({name}): {text}"
        for start_ms, role, name, text in turns
    )


def _index_parties(parties: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for party in parties:
        speaker_id = party.get("speakerId")
        if speaker_id is None:
            continue
        name = party.get("name") or UNKNOWN_NAME
        index[speaker_id] = (resolve_role(party.get("affiliation")), name)
    return index


def _format_timestamp(start_ms: int) -> str:
    total_seconds = max(int(start_ms), 0) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
