"""Run the eval suite. Requires ANTHROPIC_API_KEY in environment."""
import os
import pytest
from penny.evals.runner import run_all, print_report


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
def test_eval_suite():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    results = run_all(api_key, trials=3)
    print_report(results)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total else 0

    assert pass_rate >= 0.80, f"Eval pass rate {pass_rate:.0%} below 80% threshold"
