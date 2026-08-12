"""Tests for cross-account transfer/payment reconciliation."""
from penny.ingest.reconcile import reconcile


def _tx(date, amount, desc="", acct="1111"):
    return {
        "date": date, "amount": amount, "description": desc, "account_last4": acct,
        "is_internal": False,
    }


def test_opposite_sign_pair_tagged_internal():
    txns = [
        _tx("2024-01-05", -500.0, acct="1111"),
        _tx("2024-01-05", 500.0, acct="2222"),
    ]
    result = reconcile(txns)
    assert all(t["is_internal"] for t in result)


def test_same_account_not_matched():
    txns = [
        _tx("2024-01-05", -500.0, acct="1111"),
        _tx("2024-01-05", 500.0, acct="1111"),
    ]
    result = reconcile(txns)
    assert not any(t["is_internal"] for t in result)


def test_unrelated_transactions_not_tagged():
    txns = [
        _tx("2024-01-05", -45.0, desc="GROCERY STORE", acct="1111"),
        _tx("2024-02-20", -12.0, desc="COFFEE SHOP", acct="1111"),
    ]
    result = reconcile(txns)
    assert not any(t["is_internal"] for t in result)


def test_picks_closest_date_among_multiple_same_amount_candidates():
    # Three $500 transactions across two accounts, same amount — the payment
    # should pair with the transaction closest in date, not just the first
    # one found in list order.
    txns = [
        _tx("2024-01-01", -500.0, acct="1111"),   # far
        _tx("2024-01-20", -500.0, acct="1111"),   # close
        _tx("2024-01-21", 500.0, acct="2222"),    # the payment
    ]
    result = reconcile(txns)
    assert result[0]["is_internal"] is False   # the far one stays untouched
    assert result[1]["is_internal"] is True    # closest date match
    assert result[2]["is_internal"] is True


def test_payment_keyword_matches_same_sign():
    txns = [
        _tx("2024-01-05", -500.0, desc="ONLINE PAYMENT TO CREDIT CARD", acct="1111"),
        _tx("2024-01-06", -500.0, desc="AUTOPAY", acct="2222"),
    ]
    result = reconcile(txns)
    assert all(t["is_internal"] for t in result)
