from __future__ import annotations

from gong_mcp.formatter import format_transcript, resolve_role


def test_resolve_role_maps_affiliations():
    assert resolve_role("Internal") == "Internal"
    assert resolve_role("External") == "Customer"
    assert resolve_role("Unknown") == "Unknown"
    assert resolve_role(None) == "Unknown"


def test_format_transcript_labels_speakers_and_orders_chronologically():
    transcript = [
        {
            "speakerId": "internal-1",
            "sentences": [{"start": 25000, "text": "Got it, when did this start?"}],
        },
        {
            "speakerId": "customer-1",
            "sentences": [{"start": 12000, "text": "We've been seeing timeouts."}],
        },
    ]
    parties = [
        {"speakerId": "customer-1", "name": "Jane Doe", "affiliation": "External"},
        {"speakerId": "internal-1", "name": "Alex Smith", "affiliation": "Internal"},
    ]

    result = format_transcript(transcript, parties)

    assert result == (
        "[00:00:12] Customer (Jane Doe): We've been seeing timeouts.\n"
        "[00:00:25] Internal (Alex Smith): Got it, when did this start?"
    )


def test_format_transcript_joins_multiple_sentences_in_one_turn():
    transcript = [
        {
            "speakerId": "customer-1",
            "sentences": [
                {"start": 1000, "text": "Hi there."},
                {"start": 2000, "text": "We have a question."},
            ],
        }
    ]
    parties = [{"speakerId": "customer-1", "name": "Jane Doe", "affiliation": "External"}]

    result = format_transcript(transcript, parties)

    assert result == "[00:00:01] Customer (Jane Doe): Hi there. We have a question."


def test_format_transcript_unknown_speaker_not_in_parties():
    transcript = [{"speakerId": "ghost", "sentences": [{"start": 0, "text": "hello?"}]}]

    result = format_transcript(transcript, parties=[])

    assert result == "[00:00:00] Unknown (Unknown Speaker): hello?"


def test_format_transcript_skips_monologues_with_no_sentence_text():
    transcript = [
        {"speakerId": "customer-1", "sentences": []},
        {"speakerId": "customer-1", "sentences": [{"start": 0, "text": "  "}]},
        {"speakerId": "customer-1", "sentences": [{"start": 5000, "text": "Actual content."}]},
    ]
    parties = [{"speakerId": "customer-1", "name": "Jane Doe", "affiliation": "External"}]

    result = format_transcript(transcript, parties)

    assert result == "[00:00:05] Customer (Jane Doe): Actual content."


def test_format_transcript_formats_hour_long_timestamps():
    transcript = [{"speakerId": "s1", "sentences": [{"start": 3_725_000, "text": "hi"}]}]
    parties = [{"speakerId": "s1", "name": "Jane Doe", "affiliation": "External"}]

    result = format_transcript(transcript, parties)

    assert result.startswith("[01:02:05]")


def test_format_transcript_empty_input_returns_empty_string():
    assert format_transcript([], []) == ""
