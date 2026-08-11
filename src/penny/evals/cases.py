"""Eval test cases: outcome (exact match) + trajectory (tool-call assertions)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutcomeCase:
    """A question with a known exact answer."""
    id: str
    question: str
    expected: Any          # value to compare against
    extract: str           # key path into the agent's final numeric answer
    tolerance: float = 0.01


@dataclass
class TrajectoryCase:
    """A question that must (or must not) call a specific tool."""
    id: str
    question: str
    must_call: list[str] = field(default_factory=list)
    must_not_call: list[str] = field(default_factory=list)


OUTCOME_CASES: list[OutcomeCase] = [
    OutcomeCase("O01", "How much did I spend on dining in March 2024?",         142.50, "dining_march"),
    OutcomeCase("O02", "What was my total grocery spend in March?",             312.00, "groceries_march"),
    OutcomeCase("O03", "How much are my subscriptions costing me per month?",   89.97,  "subscriptions"),
    OutcomeCase("O04", "Total transport spend in April 2024?",                  63.40,  "transport_april"),
    OutcomeCase("O05", "How many transactions were over $100?",                 4,      "count_over_100", tolerance=0),
    OutcomeCase("O06", "Did my dining spend go up or down from March to April? By how much?",
                -28.50, "dining_delta"),
    OutcomeCase("O07", "Which merchant did I spend the most at on groceries?",  "Whole Foods", "top_grocery_merchant"),
]

TRAJECTORY_CASES: list[TrajectoryCase] = [
    TrajectoryCase("T01", "How much did I spend on dining in March?",
                   must_call=["query_sql"], must_not_call=[]),
    TrajectoryCase("T02", "Find that Uber trip in April",
                   must_call=["search_text"]),
    TrajectoryCase("T03", "Show me a bar chart of spending by category",
                   must_call=["generate_chart"]),
    TrajectoryCase("T04", "Is Netflix still $15.99 per month?",
                   must_call=["web_search"]),
    TrajectoryCase("T05", "What's my total spend excluding transfers?",
                   must_call=["query_sql"], must_not_call=[]),
]
