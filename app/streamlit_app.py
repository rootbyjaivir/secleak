"""SecLeak web app: upload a labelled flow CSV, get an integrity audit card.

Run locally:  streamlit run app/streamlit_app.py
"""
import io

import pandas as pd
import streamlit as st

from secleak.webcore import audit_frame, clean_columns, guess_benign_values, guess_label_col, guess_time_col

st.set_page_config(page_title="SecLeak", page_icon="🛡️", layout="wide")
st.title("SecLeak: is your IDS benchmark leaking?")
st.caption("Upload a labelled network-flow table. SecLeak reports duplicates, label conflicts, "
           "train/test overlap, and single-feature shortcuts. Files are processed in memory only.")

MAX_MB = 200
up = st.file_uploader("CSV file", type=["csv"])
if up is None:
    st.info("Try it with CIC-IDS2017, UNSW-NB15 or BoT-IoT CSVs, or any table with a label column.")
    st.stop()
if up.size > MAX_MB * 1024 * 1024:
    st.error(f"File larger than {MAX_MB} MB; subsample it first."); st.stop()

raw = up.getvalue()
try:
    df = clean_columns(pd.read_csv(io.BytesIO(raw), encoding="cp1252", low_memory=False, nrows=500_000))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read CSV: {e}"); st.stop()
st.write(f"Read {len(df):,} rows × {df.shape[1]} columns")

c1, c2, c3 = st.columns(3)
cols = list(df.columns)
lg = guess_label_col(df)
label_col = c1.selectbox("Label column", cols, index=cols.index(lg) if lg else 0)
vals = sorted(df[label_col].dropna().astype(str).str.strip().unique().tolist())[:200]
benign = c2.multiselect("Benign value(s)", vals, default=[v for v in guess_benign_values(df, label_col) if v in vals])
tg = guess_time_col(df)
time_col = c3.selectbox("Timestamp column (optional)", ["(none)"] + cols,
                        index=(cols.index(tg) + 1) if tg else 0)
max_rows = st.slider("Max rows to audit", 20_000, 500_000, 200_000, 20_000)

if st.button("Run audit", type="primary"):
    with st.spinner("Auditing…"):
        try:
            res = audit_frame(df, label_col, benign or None, None if time_col == "(none)" else time_col, max_rows)
        except Exception as e:  # noqa: BLE001
            st.error(str(e)); st.stop()
    for w in res["warnings"]:
        st.warning(w)
    card = res["card"]; d = card["duplication"]; ov = card["train_test_overlap"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Removable exact duplicates", f"{d['exact_removable_frac']:.1%}")
    m2.metric("Label-conflicting rows", f"{d['label_conflict_rows_frac']:.2%}")
    m3.metric("Test rows seen in train (random split)", f"{ov['random']['exact']['test_rows_in_train_frac']:.1%}")
    m4.metric("…under temporal split", f"{ov['temporal_per_class']['exact']['test_rows_in_train_frac']:.1%}")
    st.subheader("Single-feature shortcuts")
    st.dataframe(pd.DataFrame(card["top_shortcuts"]), use_container_width=True)
    st.download_button("Download audit card (markdown)", res["markdown"], "audit_card.md")
    with st.expander("Full card (JSON)"):
        st.json(card)
