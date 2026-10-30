#!/usr/bin/env python
"""Raw-CSV label audit for CIC-IDS2017 (no models needed, runs in seconds).

Question it answers: are the 'false alarms' that wreck rolling origin k=7 really benign, or is
there scan-like traffic labelled BENIGN? For each (hour, Source IP) it reports flow count,
distinct destination IPs/ports, share of tiny flows, and the label mix. Sources that scan
(many distinct dst ports) but carry BENIGN labels are candidate label errors.

usage: python scripts/label_audit.py <csv> [<csv> ...] [--min-ports 500] [--out results/label_audit.md]
"""
import argparse
from pathlib import Path

import pandas as pd


def load(path):
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    t = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")
    df["hour"] = t.dt.hour.where(t.dt.hour >= 8, t.dt.hour + 12)  # CICIDS AM/PM heuristic
    df["day"] = t.dt.date
    df["file"] = Path(path).name
    return df


def audit(df, min_ports):
    df = df.assign(tiny=df["Total Fwd Packets"] <= 2, attack=df["Label"] != "BENIGN")
    g = df.groupby(["day", "hour", "Source IP"])
    s = g.agg(flows=("Label", "size"), dst_ips=("Destination IP", "nunique"),
              dst_ports=("Destination Port", "nunique"), tiny_frac=("tiny", "mean"),
              attack_flows=("attack", "sum")).reset_index()
    s["benign_flows"] = s["flows"] - s["attack_flows"]
    s["scan_like"] = s["dst_ports"] >= min_ports
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+"); ap.add_argument("--min-ports", type=int, default=500)
    ap.add_argument("--out", default="results/label_audit.md")
    a = ap.parse_args()
    df = pd.concat([load(p) for p in a.csv], ignore_index=True)
    s = audit(df, a.min_ports)
    sc = s[s.scan_like].sort_values("benign_flows", ascending=False)
    lines = ["# Label audit (scan-like sources and their labels)", "",
             f"scan-like = >= {a.min_ports} distinct destination ports from one source in one hour", "",
             f"- total flows: {len(df):,}; BENIGN: {(df.Label=='BENIGN').sum():,}",
             f"- scan-like (hour, source) groups: {len(sc)}",
             f"- BENIGN flows inside scan-like groups: {int(sc.benign_flows.sum()):,} "
             f"({sc.benign_flows.sum()/max((df.Label=='BENIGN').sum(),1):.2%} of all benign)",
             f"- attack-labelled flows inside scan-like groups: {int(sc.attack_flows.sum()):,}", "",
             "## Top scan-like groups by BENIGN flow count", "",
             "```", sc.head(25).to_string(index=False), "```", "",
             "## Interpretation guide", "",
             "A source with hundreds of distinct ports, mostly tiny flows, and an almost-all-BENIGN",
             "label mix is port-scanning that the benchmark labels as benign. Verify against the",
             "dataset's published attack schedule before calling it a label error."]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
