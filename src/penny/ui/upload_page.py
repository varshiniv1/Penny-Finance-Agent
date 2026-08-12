"""Streamlit page: upload and ingest bank/CC statements."""
from __future__ import annotations

import streamlit as st

from penny.ingest.parser import parse_file_bytes
from penny.ingest.extractor import extract
from penny.ingest.reconcile import reconcile
from penny.ingest.enrich import enrich_batch
from penny.ui.session import get_ledger, get_fts, tx_count


def show() -> None:
    st.header("Upload Statements")
    st.caption("Supports: PDF (text or scanned/image), JPG, PNG, TIFF, CSV")

    uploaded = st.file_uploader(
        "Drop your bank and credit card statements here",
        type=["pdf", "csv", "jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        accept_multiple_files=True,
    )

    api_key = st.session_state.get("api_key", "")

    if uploaded and st.button("Process statements", type="primary"):
        ledger = get_ledger()
        fts = get_fts()
        total_new = 0

        progress = st.progress(0, text="Parsing…")
        all_transactions = []

        for i, f in enumerate(uploaded):
            progress.progress((i + 1) / (len(uploaded) + 2), text=f"Parsing {f.name}…")
            try:
                raw_rows = parse_file_bytes(f.read(), f.name)
                transactions = [t for r in raw_rows if (t := extract(r)) is not None]
                all_transactions.extend([t.to_dict() for t in transactions])
            except Exception as e:
                st.warning(f"Could not parse {f.name}: {e}")

        progress.progress((len(uploaded) + 1) / (len(uploaded) + 2), text="Reconciling…")
        reconciled = reconcile(all_transactions)

        n = ledger.upsert(reconciled)
        fts.index(reconciled)
        total_new = n

        if api_key:
            progress.progress(1.0, text="Enriching merchants…")
            try:
                stats = enrich_batch(ledger, api_key)
                st.info(
                    f"Merchant enrichment: {stats['rules']} matched by rules, "
                    f"{stats['web']} via web search, {stats['ambiguous']} ambiguous."
                )
            except Exception as e:
                st.error(f"Merchant enrichment failed: {e}")
        else:
            st.info("Add your Anthropic API key in the sidebar to enable merchant enrichment.")

        progress.empty()
        st.success(f"Loaded **{total_new}** transactions. Total in session: **{tx_count()}**.")

    # ── Session persistence ───────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Save session")
        st.caption("Download your processed transactions to restore them next time.")
        if tx_count() > 0:
            if st.button("Export transactions (Parquet)"):
                import io, tempfile, pathlib
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = pathlib.Path(tmp.name)
                get_ledger().export_parquet(tmp_path)
                st.download_button(
                    "Download penny_data.parquet",
                    data=tmp_path.read_bytes(),
                    file_name="penny_data.parquet",
                    mime="application/octet-stream",
                )

    with col2:
        st.subheader("Restore session")
        st.caption("Re-upload a previously exported Parquet file.")
        restore = st.file_uploader("Upload Parquet backup", type=["parquet"], key="restore")
        if restore and st.button("Restore"):
            import tempfile, pathlib
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp.write(restore.read())
                tmp_path = pathlib.Path(tmp.name)
            n = get_ledger().import_parquet(tmp_path)
            st.success(f"Restored. Total transactions: {n}")
