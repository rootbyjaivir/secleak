"""Per-dataset 'audit card': one JSON/markdown record summarising integrity findings."""
from __future__ import annotations

import json

import numpy as np

from ..protocols import split
from .dedup import dup_report, train_test_overlap
from .shortcut import shortcut_scan


def build_card(ds, seed: int = 0, near_sig: int = 4, shortcut_rows: int = 200_000) -> dict:
    card = {"dataset": ds.name, "n_rows": len(ds), "n_features": ds.X.shape[1],
            "has_real_timestamps": not ds.time_is_proxy, "time_kind": ds.meta.get("time_kind")}
    card["duplication"] = dup_report(ds.X, ds.y, near_sig=near_sig)
    ov = {}
    for p in ("random", "temporal_per_class"):
        tr, te = split(ds, p, seed=seed)
        ov[p] = {"exact": train_test_overlap(ds.X, tr, te, y=ds.y),
                 "near": train_test_overlap(ds.X, tr, te, y=ds.y, near_sig=near_sig)}
    card["train_test_overlap"] = ov
    sc = shortcut_scan(ds, seed=seed, max_rows=shortcut_rows)
    card["top_shortcuts"] = sc.head(5).to_dict(orient="records")
    return card


def card_markdown(card: dict) -> str:
    d, ov = card["duplication"], card["train_test_overlap"]
    L = [f"## Audit card: {card['dataset']}", "",
         f"- rows: {card['n_rows']:,}  features: {card['n_features']}",
         f"- real timestamps: **{card['has_real_timestamps']}** ({card['time_kind']})",
         f"- exact-duplicate rows removable: **{d['exact_removable_frac']:.1%}**; "
         f"near-duplicate (4 s.f.): {d['near_removable_frac']:.1%}",
         f"- label-conflicting rows (same features, different label): {d['label_conflict_rows_frac']:.2%}",
         f"- test rows already in train, random split: **{ov['random']['exact']['test_rows_in_train_frac']:.1%}** "
         f"(near: {ov['random']['near']['test_rows_in_train_frac']:.1%})",
         f"- test rows already in train, temporal split: {ov['temporal_per_class']['exact']['test_rows_in_train_frac']:.1%}",
         "", "| top single-feature shortcuts | F1 (random) | F1 (temporal) |", "|---|---|---|"]
    for r in card["top_shortcuts"]:
        L.append(f"| {r['feature']} | {r['f1_random']:.3f} | {r['f1_temporal_per_class']:.3f} |")
    return "\n".join(L) + "\n"


def save_card(card: dict, path_json: str, path_md: str):
    with open(path_json, "w") as fh:
        json.dump(card, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(path_md, "w") as fh:
        fh.write(card_markdown(card))
