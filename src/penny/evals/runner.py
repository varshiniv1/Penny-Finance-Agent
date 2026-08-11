"""Eval runner: run each case N times, track pass rate."""
from __future__ import annotations

import re
from dataclasses import dataclass

from penny.evals.fixture import build_fixture_ledger, GROUND_TRUTH
from penny.evals.cases import OUTCOME_CASES, TRAJECTORY_CASES, OutcomeCase, TrajectoryCase
from penny.agent.loop import run_turn
from penny.storage.fts import FTSIndex


@dataclass
class TrialResult:
    case_id: str
    trial: int
    passed: bool
    reason: str


def run_all(api_key: str, trials: int = 3) -> list[TrialResult]:
    results = []
    ledger = build_fixture_ledger()
    fts = FTSIndex(":memory:")

    for case in OUTCOME_CASES:
        for t in range(trials):
            history: list[dict] = []
            tool_calls: list[str] = []
            answer_text = ""

            for event in run_turn(case.question, history, ledger, fts, api_key):
                if event["type"] == "text":
                    answer_text += event["text"]
                elif event["type"] == "tool_call":
                    tool_calls.append(event["name"])

            passed, reason = _grade_outcome(case, answer_text, tool_calls)
            results.append(TrialResult(case.id, t + 1, passed, reason))

    for case in TRAJECTORY_CASES:
        for t in range(trials):
            history: list[dict] = []
            tool_calls: list[str] = []

            for event in run_turn(case.question, history, ledger, fts, api_key):
                if event["type"] == "tool_call":
                    tool_calls.append(event["name"])

            passed, reason = _grade_trajectory(case, tool_calls)
            results.append(TrialResult(case.id, t + 1, passed, reason))

    return results


def _grade_outcome(case: OutcomeCase, text: str, tool_calls: list[str]) -> tuple[bool, str]:
    if "query_sql" not in tool_calls:
        return False, "query_sql not called — agent may have estimated"

    if isinstance(case.expected, str):
        if case.expected.lower() in text.lower():
            return True, "merchant name found in answer"
        return False, f"Expected '{case.expected}' not found in answer"

    # Extract numbers from the answer
    numbers = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+\.\d{2}", text)]
    for n in numbers:
        if abs(n - case.expected) <= case.tolerance:
            return True, f"Found {n} ≈ {case.expected}"
    return False, f"Expected {case.expected}, got numbers: {numbers}"


def _grade_trajectory(case: TrajectoryCase, tool_calls: list[str]) -> tuple[bool, str]:
    for must in case.must_call:
        if must not in tool_calls:
            return False, f"Required tool '{must}' not called. Called: {tool_calls}"
    for must_not in case.must_not_call:
        if must_not in tool_calls:
            return False, f"Forbidden tool '{must_not}' was called."
    return True, "All tool assertions passed"


def print_report(results: list[TrialResult]) -> None:
    by_case: dict[str, list[TrialResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)

    print(f"\n{'Case':<6} {'Pass rate':<12} {'Details'}")
    print("-" * 60)
    total_pass = total = 0
    for case_id, trials in sorted(by_case.items()):
        n_pass = sum(1 for r in trials if r.passed)
        rate = n_pass / len(trials)
        total_pass += n_pass
        total += len(trials)
        status = "✓" if rate == 1.0 else ("~" if rate > 0 else "✗")
        reason = next((r.reason for r in trials if not r.passed), trials[0].reason)
        print(f"{status} {case_id:<5} {n_pass}/{len(trials)} ({rate:.0%})    {reason[:50]}")

    print("-" * 60)
    print(f"Overall: {total_pass}/{total} ({total_pass/total:.0%})\n")
