"""Tests for the chat page's Markdown normalization and upload-coverage
labeling — both pure functions, no Streamlit runtime needed."""
from penny.ui.chat_page import _coverage_label, _normalize_markdown


def test_downgrades_h1_heading_to_bold():
    assert _normalize_markdown("# Spending Analysis") == "**Spending Analysis**"


def test_downgrades_deep_heading_levels():
    assert _normalize_markdown("### Notes") == "**Notes**"
    assert _normalize_markdown("###### Tiny") == "**Tiny**"


def test_heading_only_at_line_start():
    # A '#' mid-sentence (e.g. a hashtag-like reference) isn't a heading.
    assert _normalize_markdown("see issue #42 for details") == "see issue #42 for details"


def test_heading_downgrade_preserves_surrounding_text():
    text = "Intro line.\n\n# Spending Analysis\n\nBody paragraph here."
    result = _normalize_markdown(text)
    assert "# Spending Analysis" not in result
    assert "**Spending Analysis**" in result
    assert "Intro line." in result
    assert "Body paragraph here." in result


def test_escapes_dollar_pairs():
    text = "Total was $11,411.67 vs $718,000 last year."
    result = _normalize_markdown(text)
    assert r"\$11,411.67" in result
    assert r"\$718,000" in result


def test_does_not_double_escape_already_escaped_dollar():
    text = r"Already escaped \$5.00"
    result = _normalize_markdown(text)
    assert result == text


def test_heading_and_dollar_together():
    text = "# Spending Analysis\n\nTotal: $11,411.67 ($718,000 payment)."
    result = _normalize_markdown(text)
    assert result.startswith("**Spending Analysis**")
    assert r"\$11,411.67" in result
    assert r"\$718,000" in result


def test_plain_text_unaffected():
    text = "Nothing special here, just a normal sentence."
    assert _normalize_markdown(text) == text


# ── Month/year coverage labeling ────────────────────────────────────────────

def test_single_file_single_month():
    assert _coverage_label([("2026-01-05", "2026-01-28")]) == "Jan 2026"


def test_single_file_contiguous_span():
    assert _coverage_label([("2026-01-05", "2026-04-28")]) == "Jan 2026 – Apr 2026"


def test_multiple_files_combine_into_one_contiguous_span():
    # Two statements, back to back — Jan and Feb — should read as one span.
    ranges = [("2026-01-05", "2026-01-31"), ("2026-02-01", "2026-02-28")]
    assert _coverage_label(ranges) == "Jan 2026 – Feb 2026"


def test_gap_is_not_reported_as_continuous():
    # Jan and Apr, nothing in between — must not claim "Jan 2026 – Apr 2026".
    ranges = [("2026-01-05", "2026-01-31"), ("2026-04-01", "2026-04-30")]
    result = _coverage_label(ranges)
    assert result != "Jan 2026 – Apr 2026"
    assert "Jan 2026" in result and "Apr 2026" in result


def test_year_boundary_is_handled():
    assert _coverage_label([("2025-12-01", "2026-01-31")]) == "Dec 2025 – Jan 2026"


def test_empty_input_returns_empty_string():
    assert _coverage_label([]) == ""


def test_rows_with_missing_dates_are_skipped():
    assert _coverage_label([(None, None), ("2026-01-05", "2026-01-28")]) == "Jan 2026"
