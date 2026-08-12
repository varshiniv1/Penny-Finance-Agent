"""Fast, API-free tests for the statement parser — no PDF/OCR libs needed,
since these exercise the pure text/CSV parsing helpers directly."""
from penny.ingest.parser import (
    _detect_statement_type,
    _dedup,
    _extract_amount,
    _extract_date,
    _map_csv_row,
    parse_file_bytes,
)


def test_bank_statement_flips_sign():
    # Bank/checking convention: withdrawal prints negative in the source text.
    # This app's convention is positive = expense, so it should flip to +45.
    assert _extract_amount("01/02 GROCERY STORE -45.00", "bank") == 45.00


def test_bank_statement_deposit_becomes_negative():
    # A plain positive deposit becomes negative (income) under this app's
    # positive=expense convention.
    assert _extract_amount("01/02 PAYCHECK 500.00", "bank") == -500.00


def test_credit_card_purchase_stays_positive():
    # CC convention: purchases print positive already — no flip needed.
    assert _extract_amount("01/02 COFFEE SHOP 4.50", "credit_card") == 4.50


def test_credit_card_payment_is_negative():
    assert _extract_amount("01/02 PAYMENT - THANK YOU -500.00", "credit_card") == -500.00


def test_credit_marker_forces_negative_regardless_of_type():
    assert _extract_amount("01/02 REFUND 12.00", "bank") == -12.00
    assert _extract_amount("01/02 REFUND 12.00", "credit_card") == -12.00


def test_detect_statement_type_requires_two_markers():
    cc_text = "Minimum Payment Due: $25.00\nCredit Limit: $5,000.00\nNew Balance: $312.10"
    assert _detect_statement_type(cc_text) == "credit_card"

    # A bank statement mentioning "payment" once shouldn't misfire.
    bank_text = "Beginning Balance: $1,000\nElectronic Withdrawals\nOnline bill payment -50.00"
    assert _detect_statement_type(bank_text) == "bank"

    assert _detect_statement_type("") == "bank"


def test_extract_date_bare_md():
    assert _extract_date("3/14 STORE 10.00") == "3/14"


def test_map_csv_row_credit_column_is_negative():
    row = {"Date": "2024-01-01", "Description": "Refund", "Debit": "", "Credit": "50.00"}
    mapped = _map_csv_row(row, "statement.csv", 1)
    assert mapped["amount"] == -50.00


def test_map_csv_row_credit_column_ignores_source_sign():
    # Even if the source file already wrote the credit as negative, a
    # dedicated Credit column always means money in for this app.
    row = {"Date": "2024-01-01", "Description": "Refund", "Debit": "", "Credit": "-50.00"}
    mapped = _map_csv_row(row, "statement.csv", 1)
    assert mapped["amount"] == -50.00


def test_dedup_keeps_same_day_same_amount_on_different_pages():
    rows = [
        {"date": "2024-01-01", "description": "COFFEE SHOP", "amount": 5.0, "page_num": 1},
        {"date": "2024-01-01", "description": "COFFEE SHOP", "amount": 5.0, "page_num": 2},
    ]
    assert len(_dedup(rows)) == 2


def test_dedup_collapses_exact_duplicate_on_same_page():
    rows = [
        {"date": "2024-01-01", "description": "COFFEE SHOP", "amount": 5.0, "page_num": 1},
        {"date": "2024-01-01", "description": "COFFEE SHOP", "amount": 5.0, "page_num": 1},
    ]
    assert len(_dedup(rows)) == 1


def test_parse_file_bytes_csv_stamps_content_hash():
    data = b"Date,Description,Amount\n2024-01-01,Coffee,5.00\n"
    rows = parse_file_bytes(data, "statement.csv")
    assert len(rows) == 1
    assert rows[0]["content_hash"]


def test_parse_file_bytes_content_hash_stable_across_rename():
    data = b"Date,Description,Amount\n2024-01-01,Coffee,5.00\n"
    rows_a = parse_file_bytes(data, "statement.csv")
    rows_b = parse_file_bytes(data, "statement (1).csv")
    assert rows_a[0]["content_hash"] == rows_b[0]["content_hash"]
