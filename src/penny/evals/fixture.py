"""Seed a frozen fixture DuckDB with known transactions for eval runs."""
from __future__ import annotations

from datetime import date, timedelta
from penny.storage.ledger import Ledger

# Ground-truth totals for the eval suite to assert against
GROUND_TRUTH = {
    "total_dining_march":       142.50,
    "total_groceries_march":    312.00,
    "total_subscriptions":       89.97,
    "total_transport_april":     63.40,
    "top_merchant":             "Whole Foods",
    "tx_count_over_100":         4,
    "march_vs_april_dining_delta": -28.50,  # April dining - March dining
}

_FIXTURE_ROWS = [
    # March — Dining
    {"date": "2024-03-05", "description": "STARBUCKS #1234",      "merchant": "Starbucks",   "category": "Dining",        "amount": 6.50,  "account_last4": "1234", "source_file": "fixture.csv", "page_num": 1},
    {"date": "2024-03-10", "description": "CHIPOTLE 987",          "merchant": "Chipotle",    "category": "Dining",        "amount": 13.00, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 1},
    {"date": "2024-03-18", "description": "DOORDASH ORDER",        "merchant": "DoorDash",    "category": "Dining",        "amount": 45.00, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 1},
    {"date": "2024-03-25", "description": "LOCAL BISTRO",          "merchant": "Local Bistro","category": "Dining",        "amount": 78.00, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 1},
    # March — Groceries
    {"date": "2024-03-02", "description": "WHOLE FOODS MKT",       "merchant": "Whole Foods", "category": "Groceries",     "amount": 156.00,"account_last4": "1234", "source_file": "fixture.csv", "page_num": 1},
    {"date": "2024-03-16", "description": "WHOLE FOODS MKT",       "merchant": "Whole Foods", "category": "Groceries",     "amount": 156.00,"account_last4": "1234", "source_file": "fixture.csv", "page_num": 2},
    # March — Subscriptions
    {"date": "2024-03-01", "description": "NETFLIX.COM",           "merchant": "Netflix",     "category": "Subscriptions", "amount": 15.99, "account_last4": "5678", "source_file": "fixture.csv", "page_num": 1},
    {"date": "2024-03-01", "description": "SPOTIFY USA",           "merchant": "Spotify",     "category": "Subscriptions", "amount": 9.99,  "account_last4": "5678", "source_file": "fixture.csv", "page_num": 1},
    {"date": "2024-03-15", "description": "ADOBE CREATIVE CLD",    "merchant": "Adobe",       "category": "Subscriptions", "amount": 54.99, "account_last4": "5678", "source_file": "fixture.csv", "page_num": 1},
    # April — Dining (total = 114.00, delta = 114.00 - 142.50 = -28.50)
    {"date": "2024-04-03", "description": "STARBUCKS #1234",       "merchant": "Starbucks",   "category": "Dining",        "amount": 7.00,  "account_last4": "1234", "source_file": "fixture.csv", "page_num": 3},
    {"date": "2024-04-12", "description": "CHIPOTLE 987",          "merchant": "Chipotle",    "category": "Dining",        "amount": 14.00, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 3},
    {"date": "2024-04-20", "description": "LOCAL BISTRO",          "merchant": "Local Bistro","category": "Dining",        "amount": 93.00, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 3},
    # April — Transport
    {"date": "2024-04-07", "description": "UBER TRIP",             "merchant": "Uber",        "category": "Transport",     "amount": 23.40, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 3},
    {"date": "2024-04-14", "description": "LYFT RIDE",             "merchant": "Lyft",        "category": "Transport",     "amount": 40.00, "account_last4": "1234", "source_file": "fixture.csv", "page_num": 3},
    # Large transactions (for "over $100" test)
    {"date": "2024-03-22", "description": "AMAZON.COM ORDER",      "merchant": "Amazon",      "category": "Shopping",      "amount": 189.99,"account_last4": "1234", "source_file": "fixture.csv", "page_num": 2},
    {"date": "2024-04-05", "description": "BEST BUY 00123",        "merchant": "Best Buy",    "category": "Shopping",      "amount": 349.00,"account_last4": "1234", "source_file": "fixture.csv", "page_num": 3},
    {"date": "2024-03-30", "description": "DELTA AIR 1234",        "merchant": "Delta",       "category": "Travel",        "amount": 412.00,"account_last4": "5678", "source_file": "fixture.csv", "page_num": 2},
    {"date": "2024-04-18", "description": "HILTON HOTEL",          "merchant": "Hilton",      "category": "Travel",        "amount": 253.00,"account_last4": "5678", "source_file": "fixture.csv", "page_num": 4},
    # Internal transfer (should be excluded from spend totals)
    {"date": "2024-03-28", "description": "TRANSFER TO SAVINGS",   "merchant": None,          "category": "Transfer",      "amount": 500.00,"account_last4": "1234", "source_file": "fixture.csv", "page_num": 2, "is_internal": True},
]


def build_fixture_ledger() -> Ledger:
    """Return an in-memory Ledger pre-seeded with fixture data."""
    import hashlib
    ledger = Ledger(":memory:", "fixture_user")
    rows = []
    for i, r in enumerate(_FIXTURE_ROWS):
        row = dict(r)
        raw = f"{row['date']}|{row['description']}|{row['amount']:.2f}|fixture|{i}"
        row["id"] = hashlib.sha1(raw.encode()).hexdigest()[:16]
        row.setdefault("account", "")
        row.setdefault("is_transfer", False)
        row.setdefault("is_refund", False)
        row.setdefault("is_internal", False)
        rows.append(row)
    ledger.upsert(rows)
    return ledger
