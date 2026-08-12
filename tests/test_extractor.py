"""Fast tests for row normalization and dedup-id stability."""
from penny.ingest.extractor import extract, _mask_account_numbers, _parse_date


def _row(**overrides):
    base = {
        "date": "01/15/2024",
        "description": "STARBUCKS #1234",
        "amount": "5.50",
        "account_last4": "1234",
        "source_file": "statement.pdf",
        "page_num": 1,
        "content_hash": "abc123",
    }
    base.update(overrides)
    return base


def test_extract_basic_row():
    t = extract(_row())
    assert t is not None
    assert t.date == "2024-01-15"
    assert t.amount == 5.50
    assert t.account_last4 == "1234"


def test_extract_rejects_unparseable_date():
    assert extract(_row(date="not a date")) is None


def test_extract_rejects_empty_description():
    assert extract(_row(description="   ")) is None


def test_is_refund_flag_from_amount_sign():
    assert extract(_row(amount="-20.00")).is_refund is True
    assert extract(_row(amount="20.00")).is_refund is False


def test_is_refund_flag_from_keyword():
    assert extract(_row(description="AMAZON REFUND", amount="20.00")).is_refund is True


def test_id_stable_across_filename_rename_same_content_hash():
    # Re-uploading the same statement under a renamed file (browser
    # auto-rename, re-export) must not double-count — same content_hash,
    # same page/date/description/amount -> same id.
    a = extract(_row(source_file="statement.pdf", content_hash="samehash"))
    b = extract(_row(source_file="statement (1).pdf", content_hash="samehash"))
    assert a.id == b.id


def test_id_differs_for_different_content():
    a = extract(_row(content_hash="hash-one"))
    b = extract(_row(content_hash="hash-two"))
    assert a.id != b.id


def test_id_falls_back_to_source_file_without_content_hash():
    row = _row()
    del row["content_hash"]
    t = extract(row)
    assert t is not None
    assert t.id


def test_mask_account_numbers():
    assert _mask_account_numbers("ACCT 123456789012") == "ACCT ****9012"


def test_parse_date_multiple_formats():
    assert _parse_date("01/15/2024") == "2024-01-15"
    assert _parse_date("2024-01-15") == "2024-01-15"
    assert _parse_date("Jan 15, 2024") == "2024-01-15"
    assert _parse_date("15-Jan-2024") == "2024-01-15"
    assert _parse_date("garbage") is None
