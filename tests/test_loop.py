"""Tests for the agent loop's history windowing — pure function, no API calls."""
from penny.agent.loop import _trim_history, _MAX_HISTORY_TURNS, _TRIM_SLACK


def _turn(n: int) -> list[dict]:
    """A user turn plus one assistant reply plus one tool-result continuation
    (role=user, list content) — same shape run_turn actually builds."""
    return [
        {"role": "user", "content": f"question {n}"},
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {n}"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "{}"}]},
    ]


def _turn_count(history: list[dict]) -> int:
    return sum(1 for m in history if m["role"] == "user" and isinstance(m["content"], str))


def test_no_trim_below_threshold():
    history = [m for n in range(_MAX_HISTORY_TURNS) for m in _turn(n)]
    before = len(history)
    _trim_history(history)
    assert len(history) == before
    assert _turn_count(history) == _MAX_HISTORY_TURNS


def test_no_trim_within_slack():
    # Right up to (but not past) the slack buffer: still no trim, so the
    # cached prefix from previous turns keeps matching.
    n_turns = _MAX_HISTORY_TURNS + _TRIM_SLACK
    history = [m for n in range(n_turns) for m in _turn(n)]
    before = list(history)
    _trim_history(history)
    assert history == before


def test_trims_once_slack_exceeded():
    n_turns = _MAX_HISTORY_TURNS + _TRIM_SLACK + 1
    history = [m for n in range(n_turns) for m in _turn(n)]
    _trim_history(history)
    assert _turn_count(history) == _MAX_HISTORY_TURNS
    # Keeps the most recent turns, drops the oldest.
    assert history[0]["content"] == f"question {n_turns - _MAX_HISTORY_TURNS}"
    assert history[-3]["content"] == f"question {n_turns - 1}"


def test_never_splits_a_tool_use_from_its_result():
    n_turns = _MAX_HISTORY_TURNS + _TRIM_SLACK + 3
    history = [m for n in range(n_turns) for m in _turn(n)]
    _trim_history(history)
    # First message after trim must be a genuine user turn, never a
    # tool_result continuation (which would leave a prior tool_use dangling).
    assert history[0]["role"] == "user"
    assert isinstance(history[0]["content"], str)
